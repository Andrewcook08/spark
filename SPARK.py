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
    report += f"🔹 **Player Name:** {player_data['Player']}\n"

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
