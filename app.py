# libraries
from multiprocessing.sharedctypes import Value
import re
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# chat initialization
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    #msg = user input
    msg = request.form["msg"]
    res = ""
    print("Input:", msg)
    if "IWTPAG" in msg:
        res = play_game(msg)
    elif re.search('^[\+\-\*\/\(\)\**\.\ 0-9]*$',msg):
        try:
            res = str(eval(msg))
        except SyntaxError:
            res = "Syntax Error"
        except ZeroDivisionError:
            res = "Math Error"
        except:
            res = "Random Error"
    else:
        ints = predict_class(msg, model)
        res = get_response(ints, intents)
    
    if '[human]' in res:
        if 'is' in msg:
            s = re.search('is',msg)
            pos = s.start() + 3
            name = msg[pos:]
            res = res.replace('[human]',name)

        elif 'am' in msg:
            s = re.search('am',msg)
            pos = s.start() + 3
            name = msg[pos:]
            res = res.replace('[human]',name)
    
    print("Output:", res)
            
    return res


# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):
    try:
        print(ints)
        tag = ints[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
    except IndexError:
        result = "Excuse me? I REALLY have no idea what are you talking about."
    except:
        result = "?"
    return result

#tic tac toe module
board = [' ' for x in range(10)]


def insertLetter(letter, pos):
    board[pos] = letter


def spaceIsFree(pos):
    return board[pos] == ' '


def nbsp(num):
    string = "";
    for i in range(num):
        string += "&nbsp;"
    return string;


def printBoard(board):
    linear_board = nbsp(3) + board[1] + nbsp(3) + '|' + nbsp(3) +  board[2] + nbsp(3) + '|' + nbsp(3) + board[3] + '<br>--------------------<br>' + nbsp(3) + board[4] + nbsp(3) + '|' + nbsp(3) + board[5] + nbsp(3) + '|' + nbsp(3) + board[6] + '<br>--------------------<br>' + nbsp(3) + board[7] + nbsp(3) + '|' + nbsp(3) + board[8] + nbsp(3) + '|' + nbsp(3) + board[9] + '<br>'
    return linear_board


def isWinner(bo, le):
    return (bo[7] == le and bo[8] == le and bo[9] == le) or (bo[4] == le and bo[5] == le and bo[6] == le) or(bo[1] == le and bo[2] == le and bo[3] == le) or(bo[1] == le and bo[4] == le and bo[7] == le) or(bo[2] == le and bo[5] == le and bo[8] == le) or(bo[3] == le and bo[6] == le and bo[9] == le) or(bo[1] == le and bo[5] == le and bo[9] == le) or(bo[3] == le and bo[5] == le and bo[7] == le)


def compMove():
    possibleMoves = [x for x, letter in enumerate(board) if letter == ' ' and x != 0]
    move = 0
    for let in ['O', 'X']:
        for i in possibleMoves:
            boardCopy = board[:]
            boardCopy[i] = let
            if isWinner(boardCopy, let):
                move = i
                return move
    cornersOpen = []
    for i in possibleMoves:
        if i in [1,3,7,9]:
            cornersOpen.append(i)
    if len(cornersOpen) > 0:
        move = selectRandom(cornersOpen)
        return move
    if 5 in possibleMoves:
        move = 5
        return move
    edgesOpen = []
    for i in possibleMoves:
        if i in [2,4,6,8]:
            edgesOpen.append(i)
    if len(edgesOpen) > 0:
        move = selectRandom(edgesOpen)
    return move


def selectRandom(li):
    import random
    ln = len(li)
    r = random.randrange(0,ln)
    return li[r]


def isBoardFull(board):
    if board.count(' ') > 1:
        return False
    else:
        return True


def play_game(msg):
    global board;
    str_move = msg[-1]
    try:
       move = int(str_move)
    except:
        return printBoard(board)

    #cannot be 0, only 1-9 is accepted
    if(move == 0):
        return printBoard(board)
    
    while not(isBoardFull(board)):
        if not(isWinner(board, 'O')):
            if spaceIsFree(move):
                insertLetter('X', move)
            else:
                return printBoard(board)
        else:
            board = [' ' for x in range(10)]
            return 'lose'
            
        if not(isWinner(board, 'X')):
            move = compMove()
            if move == 0:
                board = [' ' for x in range(10)]
                return 'tie'
            else:
                insertLetter('O', move)
                return printBoard(board)
        else:
            board = [' ' for x in range(10)]
            return 'win'
    if isBoardFull(board):
        board = [' ' for x in range(10)]
        return 'tie'
  
    
if __name__ == "__main__":
    app.run()

