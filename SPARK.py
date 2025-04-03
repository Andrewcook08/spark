import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Authentication and Google Sheets Access Functions
def authenticate_google_sheets(creds_path, sheet_name):
    """
    Authenticates the Google Sheets API using the provided credentials.
    """
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)
    
    try:
        sheet = client.open(sheet_name)
        return sheet
    except Exception as e:
        print(f"Error accessing Google Sheets: {e}")
        return None

def get_google_sheet():
    """
    Gets the Google Sheet instance for accessing data.
    """
    # Define the path to your credentials file and the Google Sheet name
    creds_path = 'Credentials.json'
    sheet_name = 'Sabermetrics'
    
    sheet = authenticate_google_sheets(creds_path, sheet_name)
    return sheet

def get_team_data(sheet, team_name):
    """
    Retrieves the data for a specific team from Google Sheets and converts it into a Pandas DataFrame.
    """
    if team_name == "MasterData":
        print(f"Skipping {team_name} as it's not necessary.")
        return pd.DataFrame()
    
    try:
        team_sheet = sheet.worksheet(team_name)
        data = team_sheet.get_all_values()
        
        # Debugging output
        print(f"DEBUG: Raw data type = {type(data)}")
        print(f"DEBUG: Raw data content (first 5 rows) = {data[:5]}")

        if not data or len(data) < 2:  # Ensure data exists beyond just headers
            print(f"Warning: No data found for {team_name}. Returning empty DataFrame.")
            return pd.DataFrame()  # Return an empty DataFrame instead of None
        
        df = pd.DataFrame(data[1:], columns=data[0])  # Convert to DataFrame
        df = df.apply(pd.to_numeric, errors='ignore')  # Convert numeric columns where possible
        
        print(f"DEBUG: Returning DataFrame for {team_name}, shape = {df.shape}")
        return df  # Ensure a DataFrame is returned

    except Exception as e:
        print(f"Error retrieving data for {team_name}: {e}")
        return pd.DataFrame()

def list_all_sheets(sheet):
    """
    Lists all available sheets in the workbook.
    """
    sheets = sheet.worksheets()  # Get all worksheets
    available_teams = [s.title for s in sheets]  # Extract titles of all sheets
    print("Available Teams:", available_teams)  # Debugging line
    return available_teams

# Scouting Report Functions
def search_player_in_team(team_data, player_name, model, scaler):
    """
    Searches for a player in a specific team and generates their scouting report.
    """
    player_found = False
    
    # Search for the player in the Hitting, Pitching, and Fielding sections
    for _, row in team_data.iterrows():
        if player_name in str(row['Player']):
            player_found = True
            return generate_player_report(row, model, scaler)  # Generate the report based on player data

    if not player_found:
        return "❌ Player not found."

def generate_player_report(player_data, model, scaler):
    """
    Generates a detailed scouting report for a player based on their data and role.
    """
    report = f"📊 Player Scouting Report:\n\n"
    player_name = player_data['Player']

    # Check if the player has data for hitting, pitching, and fielding
    is_hitter = 'AVG' in player_data and player_data['AVG'] not in [None, '']
    is_pitcher = 'ERA' in player_data and player_data['ERA'] not in [None, '']
    is_fielding = 'FLD%' in player_data and player_data['FLD%'] not in [None, '']

    if is_hitter:
        # Hitting Stats
        avg = float(player_data['AVG']) if player_data['AVG'] not in [None, ''] else 0
        obp = float(player_data['OB%']) if player_data['OB%'] not in [None, ''] else 0
        slg = float(player_data['SLG%']) if player_data['SLG%'] not in [None, ''] else 0
        ops = float(player_data['OPS']) if player_data['OPS'] not in [None, ''] else 0
        strikeouts = int(player_data['SO']) if not pd.isna(player_data['SO']) else 0
        walks = int(player_data['BB']) if not pd.isna(player_data['BB']) else 0
        sb = player_data['SB-ATT'] if player_data['SB-ATT'] not in [None, ''] else '0-0'
        hr = int(player_data['HR']) if not pd.isna(player_data['HR']) else 0
        rbi = int(player_data['RBI']) if not pd.isna(player_data['RBI']) else 0
        ab = int(player_data['AB']) if player_data['AB'] not in [None, ''] else 0

        # Calculate advanced metrics
        try:
            k_bb_ratio = strikeouts / (walks if walks != 0 else 1)
        except ZeroDivisionError:
            k_bb_ratio = float('inf')

        # Isolated Power (ISO) = SLG% - AVG
        iso = slg - avg

        # Runs Created (RC)
        sb_total = sb.split('-')
        sb_attempted = int(sb_total[1])
        sb_successful = int(sb_total[0])
        sb_percentage = sb_successful / sb_attempted if sb_attempted != 0 else 0
        rc = (rbi + (0.2 * walks) + (0.9 * sb_successful) + (0.5 * hr))  # Simplified RC formula

        # Report basic metrics
        report += f"- Batting Average (AVG): {avg:.3f}\n"
        report += f"- On-base Percentage (OB%): {obp:.3f}\n"
        report += f"- Slugging Percentage (SLG%): {slg:.3f}\n"
        report += f"- OPS: {ops:.3f}\n"
        report += f"- Strikeouts: {strikeouts}\n"
        report += f"- Walks: {walks}\n"
        report += f"- Stolen Bases (SB-ATT): {sb}\n"
        report += f"- Home Runs (HR): {hr}\n"
        report += f"- Runs Batted In (RBI): {rbi}\n"

        # Hitting performance analysis
        if avg > 0.300 and ops > 0.900:
            report += f"⚾ **Hitting Performance**: Elite hitter with excellent contact, power, and plate discipline. Can drive in runs and get on base consistently.\n"
        elif avg > 0.280 and ops > 0.800:
            report += f"⚾ **Hitting Performance**: Strong offensive contributor with solid hitting ability, power, and on-base skills.\n"
        else:
            report += f"⚾ **Hitting Performance**: Needs to improve consistency and discipline. Struggles with contact and plate approach.\n"

        # Strikeout-to-Walk Ratio Analysis
        if k_bb_ratio < 1:
            report += f"⚾ **Plate Discipline**: Strong plate discipline with more walks than strikeouts. A disciplined hitter.\n"
        else:
            report += f"⚾ **Plate Discipline**: Needs improvement in plate discipline. More strikeouts than walks indicate a tendency to chase pitches.\n"

        # Isolated Power (ISO) analysis
        if iso > 0.200:
            report += f"⚾ **Power**: Displays excellent power, regularly hitting for extra bases.\n"
        elif iso > 0.150:
            report += f"⚾ **Power**: Has good power potential but could benefit from more consistent hitting for extra bases.\n"
        else:
            report += f"⚾ **Power**: Needs to improve raw power and ability to drive the ball for extra bases.\n"

        # Stolen Bases and Aggressiveness
        if sb_percentage > 0.75:
            report += f"⚾ **Base Running**: Excellent base-running ability with a high stolen base success rate.\n"
        elif sb_percentage > 0.50:
            report += f"⚾ **Base Running**: Aggressive base runner with a decent success rate.\n"
        else:
            report += f"⚾ **Base Running**: Needs to improve base-running decision-making and success rate.\n"

        # Runs Created (RC) Analysis
        if rc > 100:
            report += f"⚾ **Offensive Impact**: Major contributor to the offense. Can create runs in a variety of ways.\n"
        elif rc > 50:
            report += f"⚾ **Offensive Impact**: Solid offensive contributor with a balanced approach.\n"
        else:
            report += f"⚾ **Offensive Impact**: Needs to improve overall offensive contribution and consistency.\n"

    if is_pitcher:
        # Pitching Stats
        era = float(player_data['ERA']) if player_data['ERA'] not in [None, ''] else 0
        report += f"- ERA: {era}\n"
        report += f"- WHIP: {player_data['WHIP']}\n"
        report += f"- Wins-Losses: {player_data['W-L']}\n"
        report += f"- Appearances-Games Started: {player_data['APP-GS']}\n"
        report += f"- Complete Games: {player_data['CG']}\n"
        report += f"- Shutouts: {player_data['SHO']}\n"
        report += f"- Saves: {player_data['SV']}\n"
        report += f"- Innings Pitched: {player_data['IP']}\n"
        report += f"- Strikeouts: {player_data['SO_Pitching']}\n"

        # Pitching analysis
        if era < 3.00:
            report += f"🔥 **Pitching Performance**: Dominates hitters with a low ERA. Shows good control and ability to manage games.\n"
        elif era < 4.00:
            report += f"🔥 **Pitching Performance**: Solid pitcher, but could improve control and avoid hard contact.\n"
        else:
            report += f"🔥 **Pitching Performance**: Needs improvement in preventing earned runs. High ERA suggests issues with command and consistency.\n"

        # WHIP analysis
        if float(player_data['WHIP']) < 1.20:
            report += f"⚾ **Pitching Control**: Excellent control with few base runners. Can dominate in high-leverage situations.\n"
        else:
            report += f"⚾ **Pitching Control**: Needs improvement in reducing walks and limiting base runners.\n"


    if "AVG" in player_data:
        report += f"🔹 **Current AVG:** {float(player_data['AVG']):.3f}\n"

    if all(col in player_data for col in ["OB%", "SLG%", "OPS"]):
        predicted_avg = predict_batting_avg(player_data, model, scaler)
        report += f"🔹 **Predicted AVG (Next Phase):** {predicted_avg}\n\n"

    # TODO add other code to add other predictions to the report here

    return report

# Train Prediction Model
def train_prediction_model(df):
    required_columns = ["AVG", "OB%", "SLG%", "OPS"]
    print(f"DEBUG: DataFrame columns = {df.columns.tolist()}")

    if not all(col in df.columns for col in required_columns):
        return None, None  # Skip training if required columns are missing
    
    df = df.dropna(subset=required_columns)
    X = df[["OB%", "SLG%", "OPS"]].astype(float)
    y = df["AVG"].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler

# Function to predict a player's future AVG
def predict_batting_avg(player_data, model, scaler):
    if model is None or scaler is None:
        return "Prediction unavailable (insufficient data)."

    try:
        features = np.array([[float(player_data["OB%"]), float(player_data["SLG%"]), float(player_data["OPS"])]])
        features_scaled = scaler.transform(features)
        predicted_avg = model.predict(features_scaled)[0]
        return f"{predicted_avg:.3f}"
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction unavailable."
    
    

# TODO add predict_"yourStatistic" here and get the code for it so that it can be called in generate_player_report

@app.route('/get_players')
def get_players():
    """
    Fetches the player names for a given team dynamically from Google Sheets.
    """
    team_name = request.args.get('team_name')
    if not team_name:
        return jsonify({"players": []})

    sheet = get_google_sheet()
    if not sheet:
        return jsonify({"players": []})

    team_data = get_team_data(sheet, team_name)
    if team_data.empty or "Player" not in team_data.columns:
        return jsonify({"players": []})  # Return an empty list if no players found

    players = team_data["Player"].dropna().tolist()  # Convert player column to list
    return jsonify({"players": players})

# Flask route
@app.route('/', methods=['GET', 'POST'])  # Allow both GET and POST
def index():
    sheet = get_google_sheet()
    if not sheet:
        return render_template("error.html", message="Failed to access Google Sheets.")

    available_teams = list_all_sheets(sheet)  # List all teams for the form

    if request.method == 'POST':  # Handle form submission
        team_name = request.form.get('team_name')
        player_name = request.form.get('player_name')

        team_data = get_team_data(sheet, team_name)
        print(f"DEBUG: Retrieved team data for {team_name}")

        if team_data.empty:
            return render_template("index.html", available_teams=available_teams, message="No data found for this team.")

        model, scaler = train_prediction_model(team_data)

        if model is None or scaler is None:
            return render_template("index.html", available_teams=available_teams, message="Model training failed due to missing data.")

        # Search for the player
        player_report = search_player_in_team(team_data, player_name, model, scaler)

        return render_template("index.html", available_teams=available_teams, player_report=player_report)

    # If it's a GET request, just render the page
    return render_template("index.html", available_teams=available_teams)

if __name__ == '__main__':
    app.run(debug=True)
