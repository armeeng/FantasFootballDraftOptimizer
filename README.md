## personal use project

Features
Data Aggregation: Scrapes the latest player projections from ESPN and fetches ADP data from Fantasy Football Calculator.

Intelligent Merging: Uses fuzzy name matching to combine the two data sources into a single, clean dataset.

Advanced Analytics: Calculates Value Over Replacement (VOR) for each player to provide a better measure of their draft value.

Live Draft GUI: An interactive Tkinter application that allows you to:

Track the draft in real-time by marking players as they are selected.

View the best players still available.

See your current team roster and positional needs.

Get an AI-powered pick recommendation when it's your turn.

Smart Recommendations: The recommendation engine simulates the rest of the draft ("ghost drafts") to suggest the player who will lead to the best possible final roster.

How It Works
The project is split into two main scripts:

ppr_draft_data.py: This script handles all the data collection. It uses Selenium to browse ESPN for player projections and requests to get ADP data. It then cleans and merges these sources, saving the final result as a .csv file.

simulate_draft.py: This script reads the generated .csv file and launches the GUI Draft Assistant. It calculates VOR and uses this data to power the recommendation engine. When you ask for a suggestion, it evaluates the top available players by projecting the optimal final roster you could build if you selected each one.
