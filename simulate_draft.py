import pandas as pd
import numpy as np
from itertools import combinations
from functools import lru_cache
import time
import tkinter as tk
from tkinter import ttk, font
import pandas as pd
import numpy as np
from itertools import combinations
from functools import lru_cache
import time
import tkinter as tk
from tkinter import ttk, font, messagebox
from tkinter.scrolledtext import ScrolledText

pd.options.display.max_rows = None

# --- Configuration ---
MY_PICK_IN_FIRST_ROUND =11
TOTAL_TEAMS = 12
TOTAL_ROUNDS = 16

df = pd.read_csv("CHANGE_ME.csv")

df = df.drop(columns=['position', 'team', 'adp', 'adp_formatted', 'times_drafted', 'high', 'low', 'stdev', 'bye', 'Unnamed: 0'])

df = df[df['Position'] != 'K'] 

df['Name'] = df['Name'].str.strip()

bye_week_map = {'Cin': 8.0, 'Atl': 8.0, 'Min': 6.0, 'Phi': 8.0, 'Det': 8.0, 'Dal': 9.0, 'SF': 9.0, 'LAR': 8.0, 'NYG': 9.0, 'LV': 12.0, 'Mia': 12.0, 'Hou': 14.0, 'Jax': 7.0, 'Ind': 9.0, 'GB': 9.0, 'TB': 14.0, 'Bal': 11.0, 'Ari': 8.0, 'LAC': 14.0, 'Buf': 9.0, 'Wsh': 9.0, 'NYJ': 8.0, 'Sea': 8.0, 'NO': 14.0, 'Car': 14.0, 'Pit': 10.0, 'KC': 12.0, 'Chi': 5.0, 'Den': 12.0, 'Ten': 10.0, 'Cle': 8.0, 'NE': 14.0}

df['bye'] = df['Team'].map(bye_week_map)
df['weekly_proj'] = df['ProjectedPoints'] / 17

ranks = [1, 5, 20, 40, 100, 150, 200, 250]
stdevs = [0.5, 1, 3, 5, 10, 15, 20, 20]

# Calculate the StDev for each player's rank using interpolation
df['StDev'] = np.interp(df['Rank'], ranks, stdevs)
df['StDev_smooth'] = (0.3 * df['Rank']**0.75).round(2)

# Round for cleaner display (optional)
df['StDev'] = df['StDev'].round(2)

print("Calculating Value Over Replacement (VOR)...")

# Step 1: Define baseline player ranks for a 12-team league
baseline_ranks = {
    'QB': 12, # The last starting QB
    'RB': 30, # A high-end RB3/Flex
    'WR': 36, # A high-end WR3/Flex
    'TE': 12  # The last starting TE
}
baseline_scores = {}

# Step 2: Find the baseline projection score for each position
for pos, rank in baseline_ranks.items():
    # Find the Nth player at the position (iloc uses 0-based index)
    baseline_player = df[df['Position'] == pos].iloc[rank - 1]
    baseline_scores[pos] = baseline_player['weekly_proj']
    print(f"Baseline for {pos}: {baseline_player['Name']} ({baseline_scores[pos]:.2f} pts/wk)")

# Step 3: Define a function to calculate VOR for a single player
def calculate_vor(player):
    pos = player['Position']
    proj = player['weekly_proj']
    # Return the player's score minus their position's baseline
    # If the position isn't in our baseline_scores map, its VOR is 0
    return proj - baseline_scores.get(pos, 0)

# Step 4: Apply the function to the DataFrame to create the 'VOR' column
df['VOR'] = df.apply(calculate_vor, axis=1)

# Ensure the VOR is rounded for cleaner display
df['VOR'] = df['VOR'].round(2)

print("VOR Calculation Complete.\n")

def simulate_logical_draft_old(player_df):
    """
    Simulates a single, logical draft where each pick number is unique.

    Args:
        player_df (pd.DataFrame): DataFrame with 'Rank' and 'StDev' columns.

    Returns:
        pd.DataFrame: The DataFrame with a new 'Pick' column,
                      sorted to reflect the draft order.
    """
    # For reproducibility, you can set a random seed
    # np.random.seed(42)

    # 1. Generate a "draft score" for each player. We DO NOT round this value yet.
    # A lower score means the player is more likely to be drafted earlier.
    draft_scores = np.random.normal(loc=player_df['Rank'], scale=player_df['StDev'])
    
    draft_df = player_df.copy()
    draft_df['DraftScore'] = draft_scores

    # 2. Sort the entire DataFrame by this score. The player with the
    # lowest score becomes the #1 pick, second lowest is #2, and so on.
    draft_board = draft_df.sort_values(by='DraftScore').reset_index(drop=True)

    # 3. Assign the final, sequential pick numbers.
    draft_board['Pick'] = draft_board.index + 1
    
    # Define columns for a clean final output
    display_cols = ['Pick', 'Name', 'Position', 'Team', 'Rank', 'StDev', 'bye', 'weekly_proj', 'VOR']
    
    return draft_board[display_cols]

def simulate_logical_draft(player_df):
    """
    Creates a draft board sorted strictly by player Rank (ADP).
    """
    # 1. Sort the DataFrame by 'Rank' to create the draft order.
    # A lower Rank means a higher pick.
    draft_board = player_df.sort_values(by='Rank').reset_index(drop=True)

    # 2. Assign the pick number based on the sorted rank.
    # The index + 1 is now the official pick number.
    draft_board['Pick'] = draft_board.index + 1
    
    # Define columns for a clean final output
    display_cols = ['Pick', 'Name', 'Position', 'Team', 'Rank', 'StDev', 'bye', 'weekly_proj', 'VOR']
    
    return draft_board[display_cols]

def get_my_picks(my_position, teams, rounds):
    """
    Calculates all of your pick numbers in a snake draft.

    Args:
        my_position (int): Your pick number in the first round (e.g., 5).
        teams (int): The total number of teams in the league (e.g., 10).
        rounds (int): The total number of rounds in the draft.

    Returns:
        list: A list of your integer pick numbers for the entire draft.
    """
    my_pick_numbers = []
    for round_num in range(1, rounds + 1):
        # Odd rounds go in forward order
        if round_num % 2 != 0:
            pick = (round_num - 1) * teams + my_position
        # Even rounds go in reverse (snake) order
        else:
            pick = (round_num - 1) * teams + (teams - my_position + 1)
        my_pick_numbers.append(pick)
    return my_pick_numbers

full_draft_board = simulate_logical_draft(df)

# 2. Calculate the specific pick numbers you will have
my_picks = get_my_picks(MY_PICK_IN_FIRST_ROUND, TOTAL_TEAMS, TOTAL_ROUNDS)
print(f"You are picking at position {MY_PICK_IN_FIRST_ROUND} in a {TOTAL_TEAMS}-team league.")
print(f"Your picks are: {my_picks}\n")


# 3. Create the map (dictionary) of available players for each of your picks
available_players_at_my_picks = {}

for pick_num in my_picks:
    # The available players are all players with a 'Pick' number greater
    # than or equal to the current pick number.
    available_df = full_draft_board[full_draft_board['Pick'] >= pick_num].copy()
    
    # Reset the index to start from 1 for easier viewing of ranks
    available_df.reset_index(drop=True, inplace=True)
    available_df.index = available_df.index + 1
    
    available_players_at_my_picks[pick_num] = available_df


def build_roster_interactive(players_per_pick):
    """
    Allows the user to manually draft their team by choosing from available players.

    At each of the user's draft picks, this function displays a list of the top
    available players and prompts the user to type in the name of the player
    they want to select. It validates the input and adds the chosen player to
    the roster.

    Args:
        players_per_pick (dict): A dictionary where keys are the user's pick numbers
                                 and values are DataFrames of players available at that pick.

    Returns:
        pd.DataFrame: A DataFrame representing the final user-drafted roster.
    """
    my_team_list = []
    drafted_player_names = set()
    my_pick_numbers = sorted(list(players_per_pick.keys()))

    print("--- Starting Interactive Draft ---")
    print("At each pick, you will be shown the top available players.")
    print("Type the full name of the player you want to draft and press Enter.\n")

    # Iterate through each of your draft picks in order
    for i, pick_num in enumerate(my_pick_numbers):
        print("----------------------------------------------------------------------")
        print(f"ROUND {i+1} - YOUR PICK #{pick_num}")
        print("----------------------------------------------------------------------")

        # Get available players and filter out those already on your team
        available_df = players_per_pick[pick_num]
        current_options = available_df[~available_df['Name'].isin(drafted_player_names)].copy()

        # Display the top 20 available players
        display_cols = ['Name', 'Position', 'Team', 'weekly_proj', 'bye', 'Rank']
        print("Top Available Players:")
        print(current_options[display_cols].head(20).to_string(index=False))
        print("-" * 30)
        print(drafted_player_names)
        # Loop to handle user input and validation
        while True:
            try:
                # Prompt user for their choice
                chosen_name = input("Enter the name of the player to draft: ")

                # Find the player in the options (case-insensitive search for robustness)
                selection = current_options[current_options['Name'].str.lower() == chosen_name.lower()]

                if not selection.empty:
                    # Player found, add to team
                    drafted_player = selection.iloc[0].to_dict()
                    drafted_player['DraftedPick'] = pick_num
                    
                    my_team_list.append(drafted_player)
                    drafted_player_names.add(drafted_player['Name'])
                    
                    print(f"\nSUCCESS: Drafted {drafted_player['Name']} ({drafted_player['Position']}) at pick {pick_num}.\n")
                    break # Exit input loop and move to the next pick
                else:
                    # Player not found
                    print("INVALID NAME. Please check the spelling and try again.")

            except Exception as e:
                print(f"An error occurred: {e}. Please try again.")

    print("--- DRAFT COMPLETE ---")
    # Convert the list of drafted players into a final DataFrame
    my_roster_df = pd.DataFrame(my_team_list)
    
    # Reorder columns for a clean final output
    final_cols = ['DraftedPick', 'Name', 'Position', 'Team', 'bye', 'weekly_proj', 'Rank', 'Pick']
    my_roster_df = my_roster_df[final_cols]
    
    return my_roster_df

def pick_for_ghost_team_bpa(available_players, ghost_roster_df):
    """
    Picks for the ghost team using a "Best Player Available" (BPA) strategy.
    It finds the player with the highest VOR and checks if they fit an open
    starting spot. If not, it moves to the next-highest VOR player and repeats.
    """

    # 1. Define the limits for our starting lineup (NOW INCLUDES D/ST)
    starter_limits = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'D/ST': 1, 'K': 1}
    flex_limit = 1
    flex_positions = {'RB', 'WR', 'TE'}

    # 2. Get the current count of players at each position on the ghost team
    current_counts = ghost_roster_df['Position'].value_counts().to_dict()

    # 3. Sort ALL available players by VOR in descending order.
    best_available_sorted = available_players.sort_values('VOR', ascending=False)

    # 4. Loop through the sorted list of best players
    for _, player in best_available_sorted.iterrows():
        pos = player['Position']

        # --- CHECK 1: Can this player fill a required starter spot? ---
        if current_counts.get(pos, 0) < starter_limits.get(pos, 0):
            return player

        # --- CHECK 2: If not, can this player fill the FLEX spot? ---
        if pos in flex_positions:
            current_flex_eligible_players = sum(current_counts.get(p, 0) for p in flex_positions)
            max_flex_starters = (
                starter_limits['RB'] + starter_limits['WR'] + starter_limits['TE'] + flex_limit
            )

            if current_flex_eligible_players < max_flex_starters:
                return player

    # --- CHECK 3: If the loop finishes, all starting spots are full ---
    if not best_available_sorted.empty:
        return best_available_sorted.iloc[0]
    else:
        return None

def build_roster_with_vor(players_per_pick, scorer_func, full_draft_board, my_picks, current_roster=[]):
    # Initialize the roster with players who have already been drafted
    my_team_list = [p.to_dict() for p in current_roster]
    drafted_player_names = set(p['Name'] for p in current_roster)

    for i, pick_num in enumerate(my_picks):
        available_df = players_per_pick[pick_num]
        best_candidate = None
        max_projected_score = -1

        print(f"ROUND {i+1} - Evaluating options for Pick #{pick_num}...")

        # --- START MODIFICATION ---
        # Get all potential candidates for this pick
        potential_candidates = available_df[~available_df['Name'].isin(drafted_player_names)]
        
        # Check current roster composition to enforce limits
        current_roster_df = pd.DataFrame(my_team_list)
        if not current_roster_df.empty:
            num_dst = (current_roster_df['Position'] == 'D/ST').sum()
            num_qb = (current_roster_df['Position'] == 'QB').sum()
            num_te = (current_roster_df['Position'] == 'TE').sum()

            # If we already have 1 D/ST, filter them out from our options
            if num_dst >= 1:
                potential_candidates = potential_candidates[potential_candidates['Position'] != 'D/ST']
            
            # If we already have 2 QBs, filter them out to avoid drafting a 3rd
            if num_qb >= 2:
                potential_candidates = potential_candidates[potential_candidates['Position'] != 'QB']

            if num_te >= 2:
                potential_candidates = potential_candidates[potential_candidates['Position'] != 'TE']

        # Now, select the top candidates from the *filtered* pool to evaluate
        candidates_to_check = potential_candidates.head(25)
        # --- END MODIFICATION ---

        for _, candidate_player in candidates_to_check.iterrows():
            # --- GHOST DRAFT LOGIC ---
            hypothetical_roster_list = my_team_list + [candidate_player.to_dict()]
            current_pick_index = my_picks.index(pick_num)
            future_picks = my_picks[current_pick_index + 1:]

            for future_pick_num in future_picks:
                ghost_team_df = pd.DataFrame(hypothetical_roster_list)
                ghost_team_names = set(ghost_team_df['Name'])
                future_pool = full_draft_board[full_draft_board['Pick'] >= future_pick_num]
                available_future_players = future_pool[~future_pool['Name'].isin(ghost_team_names)]

                if available_future_players.empty:
                    continue

                # *** USE THE SMARTER PICKING LOGIC ***
                best_available = pick_for_ghost_team_bpa(available_future_players, ghost_team_df)

                hypothetical_roster_list.append(best_available.to_dict())

            # --- Score the final, smarter ghost roster ---
            final_hypothetical_df = pd.DataFrame(hypothetical_roster_list)
            _, _, hypothetical_score = scorer_func(final_hypothetical_df)

            if hypothetical_score > max_projected_score:
                max_projected_score = hypothetical_score
                best_candidate = candidate_player

        if best_candidate is not None:
            drafted_player = best_candidate.to_dict()
            drafted_player['DraftedPick'] = pick_num
            my_team_list.append(drafted_player)
            drafted_player_names.add(best_candidate['Name'])
            print(f"  -> Selected {best_candidate['Name']} ({best_candidate['Position']}). Est. Team Score: {max_projected_score:.2f}")

    # (The rest of your function for assigning roster slots remains the same)
    # ...
    my_roster_df = pd.DataFrame(my_team_list)
    roster_slots = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'D/ST': 1, 'K': 1, 'BENCH': 6}
    final_roster_with_slots = []
    # Sort by VOR to assign starters first, ensuring the best players start.
    my_roster_df = my_roster_df.sort_values(by='VOR', ascending=False)
    for _, player in my_roster_df.iterrows():
        player_pos = player['Position']
        slot = ''
        if roster_slots.get(player_pos, 0) > 0:
            slot = player_pos
            roster_slots[player_pos] -= 1
        elif player_pos in ['RB', 'WR', 'TE'] and roster_slots['FLEX'] > 0:
            slot = 'FLEX'
            roster_slots['FLEX'] -= 1
        elif roster_slots.get('BENCH', 0) > 0:
            slot = 'BENCH'
            roster_slots['BENCH'] -= 1
        
        player['RosterSlot'] = slot
        final_roster_with_slots.append(player)
    final_df = pd.DataFrame(final_roster_with_slots).sort_values(by='DraftedPick')
    final_cols = ['DraftedPick', 'RosterSlot', 'Name', 'Position', 'Team', 'bye', 'weekly_proj', 'VOR', 'Rank', 'Pick']
    return final_df[final_cols]

def weekly_projected_points(roster):
    """
    Calculates the optimal starting lineup and projected points for each week.

    For each week of an 18-week season, this function determines the best possible
    starting lineup from the provided roster. It accounts for players on their
    bye week and selects the combination of starters that maximizes the total 
    projected points.

    Args:
        roster (pd.DataFrame): A DataFrame of the drafted team, including columns
                               for 'bye', 'Position', and 'weekly_proj'.

    Returns:
        tuple: A tuple containing two dictionaries:
        - weekly_lineups (dict): Maps each week number (int) to a DataFrame 
          representing the optimal starting lineup for that week.
        - weekly_totals (dict): Maps each week number (int) to the total 
          projected points (float) for that week's lineup.
    """
    weekly_lineups = {}
    weekly_totals = {}

    # Standard fantasy season is 18 weeks
    for week in range(1, 19):
        # 1. Get all players from your roster who are NOT on a bye week
        available_this_week = roster[roster['bye'] != week].copy()
        
        lineup_for_week = []
        players_in_lineup = set()

        # 2. Fill required starting positions (QB, RB, WR, TE, D/ST, K)
        # We iterate through each position and pick the top players based on projection.
        for pos, num_needed in [('QB', 1), ('RB', 2), ('WR', 2), ('TE', 1), ('D/ST', 1), ('K', 1)]:
            
            # Filter for players of the current position
            candidates = available_this_week[available_this_week['Position'] == pos]
            
            # Sort by highest projection and take the number needed
            best_at_pos = candidates.sort_values('weekly_proj', ascending=False).head(num_needed)
            
            # Add these players to our lineup for the week
            for _, player in best_at_pos.iterrows():
                lineup_for_week.append(player)
                players_in_lineup.add(player['Name'])

        # 3. Fill the FLEX position with the best remaining RB, WR, or TE
        # Filter for FLEX-eligible players who are not already in the starting lineup
        flex_candidates = available_this_week[
            (available_this_week['Position'].isin(['RB', 'WR', 'TE'])) &
            (~available_this_week['Name'].isin(players_in_lineup))
        ]
        
        # Sort to find the single best remaining player for the FLEX spot
        best_flex = flex_candidates.sort_values('weekly_proj', ascending=False).head(1)
        
        # Add the best FLEX player to the lineup if one exists
        if not best_flex.empty:
            lineup_for_week.append(best_flex.iloc[0])

        # 4. Finalize the lineup and calculate total points for the week
        lineup_df = pd.DataFrame(lineup_for_week)
        
        # Calculate total points, handling the edge case of an empty lineup
        weekly_total_points = lineup_df['weekly_proj'].sum() if not lineup_df.empty else 0
            
        weekly_lineups[week] = lineup_df.reset_index(drop=True)
        weekly_totals[week] = round(weekly_total_points, 2)

    season_total_proj = 0
    for k, v in weekly_totals.items():
        season_total_proj += v

    return weekly_lineups, weekly_totals, season_total_proj

# i_roster = build_roster_with_vor(available_players_at_my_picks, weekly_projected_points, full_draft_board, my_picks)
# l, t, stp = weekly_projected_points(i_roster)
# print(f"greed alg: {stp}")

# ii_roster = build_roster_interactive(available_players_at_my_picks)
# l, t, stp = weekly_projected_points(ii_roster)
# print(f"interactive alg: {stp}")

#run_live_draft_assistant(df, my_picks, weekly_projected_points, full_draft_board)

# ---------------- GUI DRAFT ASSISTANT ---------------- #
class DraftAssistantGUI:
    def __init__(self, root, df, full_draft_board, my_picks, available_players_at_my_picks):
        self.root = root
        self.df = df
        self.full_draft_board = full_draft_board
        self.my_picks = my_picks
        self.available_players_at_my_picks = available_players_at_my_picks
        
        # Track drafted players
        self.drafted_players = set()
        self.my_team = []
        self.current_pick = 1
        self.next_my_pick_index = 0
        
        self.setup_gui()
        self.update_displays()
    
    def setup_gui(self):

        # --- ADD THIS STYLING BLOCK ---
        style = ttk.Style(self.root)
        self.root.configure(bg='black')

        # Use a theme that is easy to customize
        style.theme_use('clam')

        # Configure styles for a black background and white text
        style.configure('.',
                        background='black',
                        foreground='white',
                        fieldbackground='black',
                        borderwidth=0)

        style.configure("Treeview",
                        background="black",
                        foreground="white",
                        fieldbackground="black")
        style.map('Treeview',
                  background=[('selected', '#004080')]) # A dark blue for selection

        style.configure("Treeview.Heading",
                        background="#1a1a1a", # Slightly lighter black for headings
                        foreground="white",
                        relief="flat")
        style.map("Treeview.Heading",
                  background=[('active', '#333333')])

        style.configure('TLabelframe', background='black', bordercolor="#555")
        style.configure('TLabelframe.Label', background='black', foreground='white')
        # --- END OF STYLING BLOCK ---

        self.root.title("Fantasy Football Draft Assistant")
        self.root.geometry("1400x900")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Draft status frame
        status_frame = ttk.LabelFrame(main_frame, text="Draft Status", padding="10")
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="", font=('Arial', 12, 'bold'))
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.next_pick_label = ttk.Label(status_frame, text="", font=('Arial', 10))
        self.next_pick_label.grid(row=1, column=0, sticky=tk.W)
        
        # Left panel - Player search and selection
        left_frame = ttk.LabelFrame(main_frame, text="Mark Drafted Players", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        
        # Search functionality
        search_frame = ttk.Frame(left_frame)
        search_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(1, weight=1)
        
        ttk.Label(search_frame, text="Search:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_players)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Player list
        list_frame = ttk.Frame(left_frame)
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Treeview for player list
        self.player_tree = ttk.Treeview(list_frame, columns=('Position', 'Team', 'Rank', 'VOR'), show='tree headings')
        self.player_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for player list
        player_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.player_tree.yview)
        player_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.player_tree.configure(yscrollcommand=player_scrollbar.set)
        
        # Configure columns
        self.player_tree.heading('#0', text='Player Name')
        self.player_tree.heading('Position', text='Pos')
        self.player_tree.heading('Team', text='Team')
        self.player_tree.heading('Rank', text='Rank')
        self.player_tree.heading('VOR', text='VOR')
        
        self.player_tree.column('#0', width=200)
        self.player_tree.column('Position', width=50)
        self.player_tree.column('Team', width=60)
        self.player_tree.column('Rank', width=60)
        self.player_tree.column('VOR', width=60)
        
        # Bind double-click to draft player
        self.player_tree.bind('<Double-1>', self.draft_selected_player)
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(button_frame, text="Mark as Drafted", command=self.draft_selected_player).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Undo Last", command=self.undo_last_draft).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(button_frame, text="Get My Pick Recommendation", command=self.get_recommendation, 
                  style='Accent.TButton').grid(row=0, column=2)
        
        # Right panel - Recommendations and my team
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # Recommendation frame
        rec_frame = ttk.LabelFrame(right_frame, text="Draft Recommendation", padding="10")
        rec_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        rec_frame.columnconfigure(0, weight=1)
        rec_frame.rowconfigure(0, weight=1)
        
        self.recommendation_text = ScrolledText(rec_frame, height=12, wrap=tk.WORD)
        self.recommendation_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # My team frame
        team_frame = ttk.LabelFrame(right_frame, text="My Team", padding="10")
        team_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        team_frame.columnconfigure(0, weight=1)
        team_frame.rowconfigure(0, weight=1)
        
        self.team_text = ScrolledText(team_frame, height=12, wrap=tk.WORD)
        self.team_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.populate_player_list()
    
    def populate_player_list(self):
        # Clear existing items
        for item in self.player_tree.get_children():
            self.player_tree.delete(item)
        
        # Get search filter
        search_term = self.search_var.get().lower()
        
        # Add players to tree
        for _, player in self.full_draft_board.iterrows():
            if player['Name'] in self.drafted_players:
                continue
                
            if search_term and search_term not in player['Name'].lower():
                continue
                        
            self.player_tree.insert('', 'end', text=player['Name'],
                                  values=(player['Position'], player['Team'],
                                         int(player['Rank']), f"{player['VOR']:.1f}"))
    
    def filter_players(self, *args):
        self.populate_player_list()
    
    def draft_selected_player(self, event=None):
        selection = self.player_tree.selection()
        if not selection:
            if event is None:  # Called from button, not double-click
                messagebox.showwarning("No Selection", "Please select a player to draft.")
            return
        
        item = selection[0]
        player_name = self.player_tree.item(item, 'text')
        
        # Add to drafted players
        self.drafted_players.add(player_name)
        
        # Check if this is my pick
        if self.next_my_pick_index < len(self.my_picks) and self.current_pick == self.my_picks[self.next_my_pick_index]:
            # Add to my team
            player_data = self.full_draft_board[self.full_draft_board['Name'] == player_name].iloc[0]
            self.my_team.append(player_data)
            self.next_my_pick_index += 1
            self.update_my_team_display()
        
        self.current_pick += 1
        self.update_displays()
    
    def undo_last_draft(self):
        if not self.drafted_players:
            messagebox.showinfo("Nothing to Undo", "No players have been drafted yet.")
            return
        
        # This is a simplified undo - in a real app you'd want to track the order
        last_player = list(self.drafted_players)[-1]
        self.drafted_players.remove(last_player)
        
        # Check if it was my pick
        if self.my_team and self.my_team[-1]['Name'] == last_player:
            self.my_team.pop()
            self.next_my_pick_index -= 1
            self.update_my_team_display()
        
        self.current_pick -= 1
        self.update_displays()
    
    def update_displays(self):
        self.populate_player_list()
        self.update_status()
    
    def update_status(self):
        self.status_label.config(text=f"Current Pick: {self.current_pick} | Players Drafted: {len(self.drafted_players)}")
        
        if self.next_my_pick_index < len(self.my_picks):
            next_pick = self.my_picks[self.next_my_pick_index]
            picks_away = next_pick - self.current_pick
            if picks_away == 0:
                self.next_pick_label.config(text="🔥 IT'S YOUR PICK! 🔥", foreground='red')
            else:
                self.next_pick_label.config(text=f"Your next pick: #{next_pick} ({picks_away} picks away)", foreground='blue')
        else:
            self.next_pick_label.config(text="Draft complete!", foreground='green')
    
    def update_my_team_display(self):
        self.team_text.delete(1.0, tk.END)
        
        if not self.my_team:
            self.team_text.insert(tk.END, "No players drafted yet.")
            return
        
        team_df = pd.DataFrame(self.my_team)
        
        # Position counts
        pos_counts = team_df['Position'].value_counts()
        self.team_text.insert(tk.END, "ROSTER SUMMARY:\n")
        self.team_text.insert(tk.END, f"QB: {pos_counts.get('QB', 0)} | RB: {pos_counts.get('RB', 0)} | ")
        self.team_text.insert(tk.END, f"WR: {pos_counts.get('WR', 0)} | TE: {pos_counts.get('TE', 0)} | ")
        self.team_text.insert(tk.END, f"D/ST: {pos_counts.get('D/ST', 0)}\n\n")
        
        self.team_text.insert(tk.END, "DRAFTED PLAYERS:\n")
        for i, (_, player) in enumerate(team_df.iterrows(), 1):
            self.team_text.insert(tk.END, f"{i}. {player['Name']} ({player['Position']}) - ")
            self.team_text.insert(tk.END, f"Rank: {int(player['Rank'])}, VOR: {player['VOR']:.1f}\n")
    
    def get_recommendation(self):
        if self.next_my_pick_index >= len(self.my_picks):
            messagebox.showinfo("Draft Complete", "You have completed all your picks!")
            return
        
        if self.current_pick != self.my_picks[self.next_my_pick_index]:
            messagebox.showinfo("Not Your Pick", f"It's not your turn yet. Your next pick is #{self.my_picks[self.next_my_pick_index]}")
            return
        
        self.recommendation_text.delete(1.0, tk.END)
        self.recommendation_text.insert(tk.END, "Calculating recommendation...\n")
        self.root.update()
        
        try:
            # Create updated available players dictionary
            current_available = {}
            for pick_num, players_df in self.available_players_at_my_picks.items():
                if pick_num >= self.current_pick:
                    # Filter out drafted players
                    available = players_df[~players_df['Name'].isin(self.drafted_players)].copy()
                    current_available[pick_num] = available
            
            # Get remaining picks
            remaining_picks = [p for p in self.my_picks[self.next_my_pick_index:]]
            
            if not remaining_picks:
                self.recommendation_text.delete(1.0, tk.END)
                self.recommendation_text.insert(tk.END, "No remaining picks!")
                return
            
            # Run the recommendation algorithm
            recommended_roster = build_roster_with_vor(
                current_available, 
                weekly_projected_points, 
                self.full_draft_board,
                remaining_picks,
                current_roster=self.my_team
            )
            
            # Display recommendation
            self.recommendation_text.delete(1.0, tk.END)
            
            if not recommended_roster.empty:
                next_player = recommended_roster.iloc[0]
                self.recommendation_text.insert(tk.END, "🎯 RECOMMENDED PICK:\n")
                self.recommendation_text.insert(tk.END, f"{next_player['Name']} ({next_player['Position']})\n")
                self.recommendation_text.insert(tk.END, f"Team: {next_player['Team']}, Rank: {int(next_player['Rank'])}\n")
                self.recommendation_text.insert(tk.END, f"VOR: {next_player['VOR']:.1f}, Projected: {next_player['weekly_proj']:.1f} pts/wk\n\n")
                
                # Show full recommended roster
                self.recommendation_text.insert(tk.END, "COMPLETE RECOMMENDED ROSTER:\n")
                self.recommendation_text.insert(tk.END, "-" * 50 + "\n")
                
                for _, player in recommended_roster.iterrows():
                    slot = player.get('RosterSlot', 'BENCH')
                    self.recommendation_text.insert(tk.END, f"{slot}: {player['Name']} ({player['Position']}) - ")
                    self.recommendation_text.insert(tk.END, f"VOR: {player['VOR']:.1f}\n")
                
                # Calculate total projection
                _, _, total_proj = weekly_projected_points(recommended_roster)
                self.recommendation_text.insert(tk.END, f"\nProjected Season Total: {total_proj:.1f} points")
            else:
                self.recommendation_text.insert(tk.END, "No recommendation available.")
                
        except Exception as e:
            self.recommendation_text.delete(1.0, tk.END)
            self.recommendation_text.insert(tk.END, f"Error generating recommendation: {str(e)}")


# Create and run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = DraftAssistantGUI(root, df, full_draft_board, my_picks, available_players_at_my_picks)
    root.mainloop()