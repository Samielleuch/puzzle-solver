# puzzle-solver
 this is an AI puzzle solver thats based on the game 15puzzle or as known in french 'jeu de taquin" using the A* algorithm to find the shortest path to our goal
* Ability to choose the dimension of the puzzle dynamically
* uses 2 diffrent Heuristics and shows the diffrence between them in terms of performance for this perticular case : <br/>

h1 :sum of the cases that are misplaced <br/>
h2: sum of the distance of each case to its final position
* Specify the starting position
* Specify the Goal position

## Installation

* Create a python3 virtualenv

`python3 -m venv venv`

* Source the virtual env

`source venv/bin/activate`

* Install the dependencies 

`pip install -r requirements.txt`

## Run 

`python main.py`


## Tests

Install tests dependencies 

`pip install  tox pytest pytest-cov flake8`

Run tests 

`pytest`

Launch tests using tox 

`tox`

