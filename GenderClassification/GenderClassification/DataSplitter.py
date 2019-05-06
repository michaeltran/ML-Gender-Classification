import nltk
from flair.data import Sentence
from flair.models import SequenceTagger
import textstat
from spellchecker import SpellChecker

import argparse
import xlsxwriter
import pandas as pd
import os
import re
import numpy as np
from xml.dom import minidom
import random
from random import shuffle
import multiprocessing as mp

from MinePOSPats import MinePOSPats
from Helper.NLTKPreprocessor import NLTKPreprocessor

TAGGER = SequenceTagger.load('pos')
SPELL = SpellChecker()
SPELL.word_frequency.load_words(["'s", "'m", "'re", "'ll" , "'ve", "'t", "'d"])
SPELL.word_frequency.load_words(['hulu', 'google', 'smartphone', 'iphone', 'immersive', 'xbox', 'blog', 'obama', 'facebook', 'itunes', 'gmail', 'espn', 'youtube', 'adsense', 'pikachu', 'etsy', 
                                 'wikipedia', 'starbucks', 'jetblue', 'webcam', 'traveler', 'retweet', 'website', 'favors', 'deadspin', 'huffington', 'wordpress', 'linkedin', 'website', 'mozilla', 
                                 'firefox', 'linux', 'http', 'evanescence', 'dvd', 'krispy', 'kreme', 'donut', '3d', 'mp3', 'jpg', 'photobucket', 'suv', 'saudia', 'vevo', 'popsicle', 'voicemail',
                                 'sophomore', 'hotmail', 'fastmail', 'morrowind', 'fanboy', 'fandom', 'resize', 'ie5', 'ie6', 'timeline', 'newbie', 'vigor', 'netflix', 'anime', 'html', 'xml', 'savior',
                                 'walgreens', 'ebay', 'myspace', 'lasik', 'wii', 'uber', 'petco', 'sweatpants', 'jello', 'malware', 'wiki', 'ubuntu', 'savor', 'nerdy', 'flavors', 'jalapenos', 'guestbook',
                                 'runescape', 'ragnarok', 'gamer', '2d', '3d', 'neverwinter', 'paintball', 'urllink', 'internship', 'warcraft', 'simcity', 'sxsw', 'paralyzed', 'telus', 'shrek', 'avatar',
                                 'poptarts', 'bioshock', 'endeavor', 'walmart', 'multiplayer', 'advil', 'ffix', 'funimation', 'fullmetal'])
SPELL.word_frequency.load_words(['dvds', 'blogs', 'blogging', 'blogger', 'bloggers', 'retweets', 'iphones', 'tweets', 'websites', 'resizes', 'resizing', 'geeks'])
NLTK_PREPROCESSOR = NLTKPreprocessor(True)

POS_DICTIONARY = {}

def SplitData():
    ## Get Command-line Arguments #################
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='data/blog-gender-dataset.xlsx', help='')
    parser.add_argument('-m', '--mine', default=True, help ='')
    opts = parser.parse_args()
    ###############################################

    LoadPOSExcel('data/pos.xlsx')

    ## Build the Training Set and Testing Set #####
    names = ['Text', 'Classification']
    df = pd.read_excel(opts.data, header=None, names=names, usecols="A,B")

    data_male_text = []
    data_female_text = []

    training_data_text = []
    training_data_classification = []
    testing_data_text = []
    testing_data_classification = []

    # Prepare and Sanitize data - -1 = Female, 1 = Male
    for i in range(len(df['Text'])):
        text = df['Text'][i]

        classification = df['Classification'][i]
        if text == text and classification == classification:
            text = CleanText(text)

            classification = classification.strip().upper()
            if classification == 'M':
                data_male_text.append(text)
            elif classification == 'F':
                data_female_text.append(text)
            else:
                print('Classification Error: %s is not defined.' % (classification))
                return

    #min = 32714
    #max = 74
    #for text in data_male_text + data_female_text:
    #    if len(text) > max:
    #        max = len(text)
    #    if len(text) < min:
    #        min = len(text)

    blog_data = []
    blog_data_classification = []
    blog_male_data, blog_female_data = GetBlogAuthorshipCorpusData('data/blogs')
    for i in range(min(len(blog_male_data), len(blog_female_data))):
        blog_data.append(blog_male_data[i])
        blog_data_classification.append(1)
        blog_data.append(blog_female_data[i])
        blog_data_classification.append(-1)

    ##data_male_text = data_male_text + blog_male_data
    ##data_female_text = data_female_text + blog_female_data

    GetPOSTags(data_male_text)
    WritePOSToExcel('data/pos.xlsx')
    GetPOSTags(data_female_text)
    WritePOSToExcel('data/pos.xlsx')
    GetPOSTags(blog_data)
    WritePOSToExcel('data/pos.xlsx')

    # POS Pattern Mining
    if opts.mine == True:
        pos_list = []
        for text in data_male_text:
            pos_list.append(GetPOSTag(text))
        for text in data_female_text:
            pos_list.append(GetPOSTag(text))

        mine_obj = MinePOSPats(pos_list, 0.3, 0.2)
        pos_pats = mine_obj.MinePOSPats()

        # Write POS Patterns to Text
        with open('data/POSPatterns.txt', 'w') as file:
            patterns = []
            for pos_pat in pos_pats:
                pattern = ' '.join(pos_pat)
                patterns.append(pattern)
            file.write('\n'.join(patterns))

    shuffle(data_male_text)
    shuffle(data_female_text)

    #c = list(zip(blog_data, blog_data_classification))
    #shuffle(c)
    #blog_data, blog_data_classification = zip(*c)

    # Split out training dataset
    len_data = 0
    len_data = min(int(len(data_male_text)), int(len(data_female_text)))
    len_training_data = int(len_data * (8/10) * 2)
    for i in range(len_training_data):
        if i % 2 == 1:
            # Male
            training_data_text.append(data_male_text[0])
            training_data_classification.append(1)
            del data_male_text[0]
        else:
            # Female
            training_data_text.append(data_female_text[0])
            training_data_classification.append(-1)
            del data_female_text[0]

    # Split out testing dataset
    for i in range(min(len(data_male_text), len(data_female_text))):
        testing_data_text.append(data_male_text[0])
        testing_data_classification.append(1)
        del data_male_text[0]
        testing_data_text.append(data_female_text[0])
        testing_data_classification.append(-1)
        del data_female_text[0]

    ## Word Pattern Mining
    #if opts.mine == True:
    #    mine_obj = MineWordPats(training_data_text + testing_data_text + data_male_text + data_female_text, 0.05, 0.05)
    #    pos_pats = mine_obj.MineWordPats()

    #    # Write Word Patterns to Text
    #    with codecs.open('data/WordPatterns.txt', 'w', encoding='utf8') as file:
    #        patterns = []
    #        for pos_pat in pos_pats:
    #            pattern = ' '.join(pos_pat)
    #            patterns.append(pattern)
    #        file.write('\n'.join(patterns))

    print("Saving Unlabled Data")

    # Save "Unlabled" Data
    WriteToExcel('data/unlabeled_data.xlsx', blog_data, blog_data_classification)
    #p1 = Process(target=WriteToExcel, args=('data/unlabeled_data.xlsx', blog_data, blog_data_classification))
    #p1.start()

    print("Saving Training Data")

    # Save Training Data
    WriteToExcel('data/train_data.xlsx', training_data_text, training_data_classification)
    #p2 = Process(target=WriteToExcel, args=('data/train_data.xlsx', training_data_text, training_data_classification))
    #p2.start()

    print("Saving Test Data")

    # Save Testing Data
    WriteToExcel('data/test_data.xlsx', testing_data_text, testing_data_classification)
    #p3 = Process(target=WriteToExcel, args=('data/test_data.xlsx', testing_data_text, testing_data_classification))
    #p3.start()

    #p1.join()
    #p2.join()
    #p3.join()

    ###############################################
    print("Completed")

def WriteToExcel(path, data_text, data_classification):
    with xlsxwriter.Workbook(path, {'strings_to_urls': False}) as workbook:
        worksheet = workbook.add_worksheet();
        row = 0;
        col = 0;
        worksheet.write(row, col, 'Classification'); col += 1;
        worksheet.write(row, col, 'Text'); col += 1;
        worksheet.write(row, col, 'TokenizedText'); col += 1;
        worksheet.write(row, col, 'TokenizedText2'); col += 1;
        worksheet.write(row, col, 'POS'); col += 1;
        worksheet.write(row, col, 'TaggedPOS'); col += 1;
        worksheet.write(row, col, 'WordCount'); col += 1;
        worksheet.write(row, col, 'Length'); col += 1;
        worksheet.write(row, col, 'F-Measure'); col += 1;
        worksheet.write(row, col, 'GPF1'); col += 1;
        worksheet.write(row, col, 'GPF2'); col += 1;
        worksheet.write(row, col, 'GPF3'); col += 1;
        worksheet.write(row, col, 'GPF4'); col += 1;
        worksheet.write(row, col, 'GPF5'); col += 1;
        worksheet.write(row, col, 'GPF6'); col += 1;
        worksheet.write(row, col, 'GPF7'); col += 1;
        worksheet.write(row, col, 'GPF8'); col += 1;
        worksheet.write(row, col, 'GPF9'); col += 1;
        worksheet.write(row, col, 'GPF10'); col += 1;
        worksheet.write(row, col, 'FA1'); col += 1;
        worksheet.write(row, col, 'FA2'); col += 1;
        worksheet.write(row, col, 'FA3'); col += 1;
        worksheet.write(row, col, 'FA4'); col += 1;
        worksheet.write(row, col, 'FA5'); col += 1;
        worksheet.write(row, col, 'FA6'); col += 1;
        worksheet.write(row, col, 'FA7'); col += 1;
        worksheet.write(row, col, 'FA8'); col += 1;
        worksheet.write(row, col, 'FA9'); col += 1;
        worksheet.write(row, col, 'FA10'); col += 1;
        worksheet.write(row, col, 'FA11'); col += 1;
        worksheet.write(row, col, 'FA12'); col += 1;
        worksheet.write(row, col, 'FA13'); col += 1;
        worksheet.write(row, col, 'FA14'); col += 1;
        worksheet.write(row, col, 'FA15'); col += 1;
        worksheet.write(row, col, 'FA16'); col += 1;
        worksheet.write(row, col, 'FA17'); col += 1;
        worksheet.write(row, col, 'FA18'); col += 1;
        worksheet.write(row, col, 'FA19'); col += 1;
        worksheet.write(row, col, 'FA20'); col += 1;
        worksheet.write(row, col, 'FA21'); col += 1;
        worksheet.write(row, col, 'FA22'); col += 1;
        worksheet.write(row, col, 'FA23'); col += 1;
        worksheet.write(row, col, 'LE_C'); col += 1;
        worksheet.write(row, col, 'TS'); col += 1;
        worksheet.write(row, col, 'WordsMispelled'); col += 1;
        row += 1;
        for i in range(len(data_text)):
            col = 0;
            text = data_text[i];

            tokenized_text = GetTokenizedText(text) # Lemmatized and Stopword Removed
            tokenized_text_2 = GetTokenizedText2(text) # Regular Tokenization

            pos = GetPOS(text); # POS
            tagged_pos = GetTaggedPOS(text); # WORD_POS
            fmeasure = GetFMeasure(text);

            gpf = GetGenderPreferentialFeatures(text);
            fa = GetFactorAnalysis(text);
            le_c, ts = GetTextStatInfo(text)
            words_misspelled = GetNumberOfIncorrectSpelling(text)

            worksheet.write(row, col, data_classification[i]); col += 1;
            worksheet.write(row, col, text); col += 1;
            worksheet.write(row, col, tokenized_text); col += 1;
            worksheet.write(row, col, tokenized_text_2); col += 1;
            worksheet.write(row, col, pos); col += 1;
            worksheet.write(row, col, tagged_pos); col += 1;
            worksheet.write(row, col, len(TextTokenizer(text))); col += 1;
            worksheet.write(row, col, len(text)); col += 1;
            worksheet.write(row, col, fmeasure); col += 1;
            worksheet.write(row, col, gpf[0]); col += 1;
            worksheet.write(row, col, gpf[1]); col += 1;
            worksheet.write(row, col, gpf[2]); col += 1;
            worksheet.write(row, col, gpf[3]); col += 1;
            worksheet.write(row, col, gpf[4]); col += 1;
            worksheet.write(row, col, gpf[5]); col += 1;
            worksheet.write(row, col, gpf[6]); col += 1;
            worksheet.write(row, col, gpf[7]); col += 1;
            worksheet.write(row, col, gpf[8]); col += 1;
            worksheet.write(row, col, gpf[9]); col += 1;
            worksheet.write(row, col, fa[0]); col += 1;
            worksheet.write(row, col, fa[1]); col += 1;
            worksheet.write(row, col, fa[2]); col += 1;
            worksheet.write(row, col, fa[3]); col += 1;
            worksheet.write(row, col, fa[4]); col += 1;
            worksheet.write(row, col, fa[5]); col += 1;
            worksheet.write(row, col, fa[6]); col += 1;
            worksheet.write(row, col, fa[7]); col += 1;
            worksheet.write(row, col, fa[8]); col += 1;
            worksheet.write(row, col, fa[9]); col += 1;
            worksheet.write(row, col, fa[10]); col += 1;
            worksheet.write(row, col, fa[11]); col += 1;
            worksheet.write(row, col, fa[12]); col += 1;
            worksheet.write(row, col, fa[13]); col += 1;
            worksheet.write(row, col, fa[14]); col += 1;
            worksheet.write(row, col, fa[15]); col += 1;
            worksheet.write(row, col, fa[16]); col += 1;
            worksheet.write(row, col, fa[17]); col += 1;
            worksheet.write(row, col, fa[18]); col += 1;
            worksheet.write(row, col, fa[19]); col += 1;
            worksheet.write(row, col, fa[20]); col += 1;
            worksheet.write(row, col, fa[21]); col += 1;
            worksheet.write(row, col, fa[22]); col += 1;
            worksheet.write(row, col, le_c); col += 1;
            worksheet.write(row, col, ts); col += 1;
            worksheet.write(row, col, words_misspelled); col += 1;

            row += 1;
    return

def GetTokenizedText(text):
    tokens = NLTK_PREPROCESSOR.TokenizeText(text)
    return ' '.join(tokens)

def GetTokenizedText2(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

def GetPOS(text):
    pos_text = GetPOSTag(text)
    return ' '.join(pos_text)

def GetTaggedPOS(text):
    tagged_text = []

    pos_text = GetPOSTag(text)
    preprocessed_text = PreprocessForPOS(text)
    sentence = Sentence(preprocessed_text, use_tokenizer=True)

    for i in range(len(pos_text)):
        tagged_text.append(sentence[i].text + '_' + pos_text[i])

    return ' '.join(tagged_text)

def GetFMeasure(text):
    tagged = GetPOSTag(text)

    freq = {}
    freq['noun'] = 0
    freq['adj'] = 0
    freq['prep'] = 0
    freq['art'] = 0
    freq['pron'] = 0
    freq['verb'] = 0
    freq['adv'] = 0
    freq['int'] = 0

    count = 0
    for i in range(len(tagged)):
        pos = tagged[i]
        if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
            freq['noun'] += 1
        elif pos in ['JJ', 'JJR', 'JJS']:
            freq['adj'] += 1
        elif pos in ['IN']:
            freq['prep'] += 1
        elif pos in ['DET', 'DT', 'PDT', 'WDT']:
            freq['art'] += 1
        elif pos in ['PRP', 'PRP$', 'WP', 'WP$']:
            freq['pron'] += 1
        elif pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            freq['verb'] += 1
        elif pos in ['RB', 'RBR', 'RBS', 'WRB']:
            freq['adv'] += 1
        elif pos in ['UH']:
            freq['int'] += 1

        if pos not in ['$', "'", '(', ')', ',', '-', '.', ':', 'SYM', "''", '``']:
            count += 1

    for key in freq:
        freq[key] = (freq[key] / count) * 100

    fmeasure = 0.5 * ( (freq['noun'] + freq['adj'] + freq['prep'] + freq['art']) - (freq['pron'] + freq['verb'] + freq['adv'] + freq['int']) + 100 )

    return fmeasure

def GetGenderPreferentialFeatures(text):
    f = []
    for i in range(10):
        f.append(0)
    for word in TextTokenizer(text):
        word = word.lower()
        if word.endswith(('able')):
            f[0] += 1
        elif word.endswith(('al')):
            f[1] += 1
        elif word.endswith(('ful')):
            f[2] += 1
        elif word.endswith(('ible')):
            f[3] += 1
        elif word.endswith(('ic')):
            f[4] += 1
        elif word.endswith(('ive')):
            f[5] += 1
        elif word.endswith(('less')):
            f[6] += 1
        elif word.endswith(('ly')):
            f[7] += 1
        elif word.endswith(('ous')):
            f[8] += 1
        if word in ['sorry', 'penitent', 'contrite', 'repentant', 'remorseful', 'regretful', 'compunctious', 'touched', 'melted', 'sorrowful', 'apologetic', 'softened'
                      'sad', 'greived', 'mournful']:
            f[9] += 1
    return f

def GetFactorAnalysis(text):
    f = []
    words_in_factor = []
    # Conversation
    words_in_factor.append(['know', 'people', 'think', 'person', 'tell', 'feel', 'friends', 'talk', 'new', 'talking', 'mean', 'ask', 'understand', 
                    'feelings', 'care', 'thinking', 'friend', 'relationship', 'realize', 'question', 'answer', 'saying'])
    # AtHome
    words_in_factor.append(['woke', 'home', 'sleep', 'today', 'eat', 'tired', 'wake', 'watch', 'watched', 'dinner', 'ate', 'bed', 'day', 'house', 'tv', 'early', 'boring', 'yesterday', 'watching', 'sit'])
    # Family
    words_in_factor.append(['years', 'family', 'mother', 'children', 'father', 'kids', 'parents', 'old', 'year', 'child', 'son', 'married', 'sister', 'dad', 'brother', 'moved', 'age', 'young', 
                            'months', 'three', 'wife', 'living', 'college', 'four', 'high', 'five', 'died', 'six', 'baby', 'boy', 'spend', 'christmas'])
    # Time
    words_in_factor.append(['friday', 'saturday', 'weekend', 'week', 'sunday', 'night', 'monday', 'tuesday', 'thursday', 'wednesday', 'morning', 'tomorrow', 'tonight', 'evening', 'days', 
                            'afternoon', 'weeks', 'hours', 'july', 'busy', 'meeting', 'hour', 'month', 'june'])
    # Work
    words_in_factor.append(['work', 'working', 'job', 'trying', 'right', 'met', 'figure', 'meet', 'start', 'better', 'starting', 'try', 'worked', 'idea'])
    # PastActions
    words_in_factor.append(['said', 'asked', 'told', 'looked', 'walked', 'called', 'talked', 'wanted', 'kept', 'took', 'sat', 'gave', 'knew', 'felt', 'turned', 'stopped', 'saw', 'ran', 'tried', 
                            'picked', 'left', 'ended'])
    # Games
    words_in_factor.append(['game', 'games', 'team', 'win', 'play', 'played', 'playing', 'won', 'season', 'beat', 'final', 'two', 'hit', 'first', 'video', 'second', 'run', 'star', 'third', 'shot', 
                            'table', 'round', 'ten', 'chance', 'club', 'big', 'straight'])
    # Internet
    words_in_factor.append(['site', 'email', 'page', 'please', 'website', 'web', 'post', 'link', 'check', 'blog', 'mail', 'information', 'free', 'send', 'comments', 'comment', 'using', 
                            'internet', 'online', 'name', 'service', 'list', 'computer', 'add', 'thanks', 'update', 'message'])
    # Location
    words_in_factor.append(['street', 'place', 'town', 'road', 'city', 'walking', 'trip', 'headed', 'front', 'car', 'beer', 'apartment', 'bus', 'area', 'park', 'building', 'walk', 'small', 'places', 
                            'ride', 'driving', 'looking', 'local', 'sitting', 'drive', 'bar', 'bad', 'standing', 'floor', 'weather', 'beach', 'view'])
    # Fun
    words_in_factor.append(['fun', 'im', 'cool', 'mom', 'summer', 'awesome', 'lol', 'stuff', 'pretty', 'ill', 'mad', 'funny', 'weird'])
    # Food/Clothes
    words_in_factor.append(['food', 'eating', 'weight', 'lunch', 'water', 'hair', 'life', 'white', 'wearing', 'color', 'ice', 'red', 'fat', 'body', 'black', 'clothes', 'hot', 'drink', 'wear', 
                            'blue', 'minutes', 'shirt', 'green', 'coffee', 'total', 'store', 'shopping'])
    # Poetic
    words_in_factor.append(['eyes', 'heart', 'soul', 'pain', 'light', 'deep', 'smile', 'dreams', 'dark', 'hold', 'hands', 'head', 'hand', 'alone', 'sun', 'dream', 'mind', 'cold', 'fall', 'air', 
                            'voice', 'touch', 'blood', 'feet', 'words', 'hear', 'rain', 'mouth'])
    # Books/Movies
    words_in_factor.append(['book', 'read', 'reading', 'books', 'story', 'writing', 'written', 'movie', 'stories', 'movies', 'film', 'write', 'character', 'fact', 'thoughts', 
                            'title', 'short', 'take', 'wrote'])
    # Religion
    words_in_factor.append(['god', 'jesus', 'lord', 'church', 'earth', 'world', 'word', 'lives', 'power', 'human', 'believe', 'given', 'truth', 'thank', 'death', 'evil', 'own', 'peace', 
                            'speak', 'bring', 'truly'])
    # Romance
    words_in_factor.append(['forget', 'forever', 'remember', 'gone', 'true', 'face', 'spent', 'times', 'love', 'cry', 'hurt', 'wish', 'loved'])
    # Swearing
    words_in_factor.append(['shit', 'fuck', 'fucking', 'ass', 'bitch', 'damn', 'hell', 'sucks', 'stupid', 'hate', 'drunk', 'crap', 'kill', 'guy', 'gay', 'kid', 'sex', 'crazy', 'cunt', 'nigger', 'nigga', 
                            'asshole', 'pussy', 'dick', 'dickhead', 'faggot', 'fag'])
    # Politics
    words_in_factor.append(['bush', 'president', 'iraq', 'kerry', 'war', 'american', 'political', 'states', 'america', 'country', 'government', 'john', 'national', 'news', 'state', 'support', 
                            'issues', 'article', 'michael', 'bill', 'report', 'public', 'issue', 'history', 'party', 'york', 'law', 'major', 'act', 'fight', 'poor'])
    # Music
    words_in_factor.append(['music', 'songs', 'song', 'band', 'cd', 'rock', 'listening', 'listen', 'show', 'favorite', 'radio', 'sound', 'heard', 'shows', 'sounds', 'amazing', 'dance'])
    # School
    words_in_factor.append(['school', 'teacher', 'class', 'study', 'test', 'finish', 'english', 'students', 'period', 'paper', 'pass'])
    # Business
    words_in_factor.append(['system', 'based', 'process', 'business', 'control', 'example', 'personal', 'experience', 'general'])
    # Positive
    words_in_factor.append(['absolutely', 'abundance', 'ace', 'active', 'admirable', 'adore', 'agree', 'amazing', 'appealing', 'attraction', 'bargain', 'beaming', 'beautiful', 'best', 'better', 
                            'boost', 'breakthrough', 'breeze', 'brilliant', 'brimming', 'charming', 'clean', 'clear', 'colorful', 'compliment', 'confidence', 'cool', 'courteous', 'cuddly', 
                            'dazzling', 'delicious', 'delightful', 'dynamic', 'easy', 'ecstatic', 'efficient', 'enhance', 'enjoy', 'enormous', 'excellent', 'exotic', 'expert', 'exquisite', 
                            'flair', 'free', 'generous', 'genius', 'great', 'graceful', 'heavenly', 'ideal', 'immaculate', 'impressive', 'incredible', 'inspire', 'luxurious', 'outstanding', 
                            'royal', 'speed', 'splendid', 'spectacular', 'superb', 'sweet', 'sure', 'supreme', 'terrific', 'treat', 'treasure', 'ultra', 'unbeatable', 'ultimate', 'unique', 'wow', 'zest'])
    # Negative
    words_in_factor.append(['wrong', 'stupid', 'bad', 'evil', 'dumb', 'foolish', 'grotesque', 'harm', 'fear', 'horrible', 'idiot', 'lame', 'mean', 'poor', 'heinous', 'hideous', 'deficient', 
                            'petty', 'awful', 'hopeless', 'fool', 'risk', 'immoral', 'risky', 'spoil', 'spoiled', 'malign', 'vicious', 'wicked', 'fright', 'ugly', 'atrocious', 'moron', 'hate', 
                            'spiteful', 'meager', 'malicious', 'lacking'])
    # Emotion
    words_in_factor.append(['aggressive', 'alienated', 'angry', 'annoyed', 'anxious', 'careful', 'cautious', 'confused', 'curious', 'depressed', 'determined', 'disappointed', 'discouraged', 
                            'disgusted', 'ecstatic', 'embarrassed', 'enthusiastic', 'envious', 'excited', 'exhausted', 'frightened', 'frustrated', 'guilty', 'happy', 'helpless', 'hopeful', 
                            'hostile', 'humiliated', 'hurt', 'hysterical', 'innocent', 'interested', 'jealous', 'lonely', 'mischievous', 'miserable', 'optimistic', 'paranoid', 'peaceful', 
                            'proud', 'puzzled', 'regretful', 'relieved', 'sad', 'satisfied', 'shocked', 'shy', 'sorry', 'surprised', 'suspicious', 'thoughtful', 'undecided', 'withdrawn'])

    for i in range(len(words_in_factor)):
        f.append(0)
    for word in TextTokenizer(text):
        word = word.lower()
        for i in range(len(words_in_factor)):
            if word in words_in_factor[i]:
                f[i] += 1
    return f

def GetBlogAuthorshipCorpusData(path):
    #for file_name in os.listdir('data/blogs'):
    #    cleaned_file = ''
    #    with open('data/blogs/' + file_name) as file:
    #        for line in file:

    male_data = []
    female_data = []

    for file_name in os.listdir(path):
        gender = re.findall("[0-9]*\.(.*)\.[0-9]+\..*\.xml", file_name)[0]
        try:
            mydoc = minidom.parse(path + '/' + file_name)
            posts = mydoc.getElementsByTagName('post')
            #print(posts[1].firstChild.data)

            for post in posts:
                text = post.firstChild.data
                text = text.replace('\n', '').replace('\t', '').strip()
                text = CleanText(text)
                if len(text) < 100 or len(text) > 32700:
                    continue
                if text.startswith("="):
                    continue
                if gender == 'male':
                    male_data.append(text)
                elif gender == 'female':
                    female_data.append(text)
                else:
                    print(gender)

            #entire_text = ''

            #for post in posts:
            #    text = post.firstChild.data
            #    text = text.replace('\n', '').replace('\t', '').strip()
            #    entire_text = entire_text + text + '\n'

            #if gender == 'male':
            #    male_data.append(entire_text)
            #elif gender == 'female':
            #    female_data.append(entire_text)
        except:
            test = 0
            #print("Skipped: " + file_name)

    return male_data, female_data

def PreprocessForPOS(text):
    new_text = text
    new_text = new_text.replace(',', ', ')
    new_text = new_text.replace('.', '. ')
    return new_text

def GetPOSTag(d_words):
    if d_words in POS_DICTIONARY:
        return POS_DICTIONARY[d_words]
    else:
        d = []

        preprocessed_text = PreprocessForPOS(d_words)
        sentence = Sentence(preprocessed_text, use_tokenizer=True)

        #for token in sentence:
        #    text = token.text
        #    #print(SPELL.candidates(text))
        #    correction = SPELL.correction(text)
        #    if text != correction:
        #        token.text = correction

        TAGGER.predict(sentence)
        for token in sentence:
            pos = token.get_tag('pos').value
            d.append(pos)

        POS_DICTIONARY[d_words] = d
        print(len(POS_DICTIONARY))
        return d

def GetPOSTags(D_words):
    i = 1
    for d_words in D_words:
        GetPOSTag(d_words)
        if i % 2000 == 0:
            print('Saving Checkpoint')
            WritePOSToExcel('data/pos.xlsx')
    return

#def GetPOSTagParallel(d_words):
#    print('sweg')
#    #d = []

#    #print(1)
#    #preprocessed_text = PreprocessForPOS(d_words)
#    #sentence = Sentence(preprocessed_text, use_tokenizer=True)
#    #print(2)
#    #for token in sentence:
#    #    print(3)
#    #    text = token.text
#    #    #print(SPELL.candidates(text))
#    #    correction = SPELL.correction(text)
#    #    if text != correction:
#    #        token.text = correction
#    #    print(4)
#    #print(5)
#    #TAGGER.predict(sentence)
#    #for token in sentence:
#    #    pos = token.get_tag('pos').value
#    #    d.append(pos)

#    #return (d_words, d)
#    return (1, 2)

#def GetPOSTagsParallel(D_words):
#    pool = mp.Pool(2)
#    def CollectResults(result):
#        print('derp')
#        POS_DICTIONARY[result[0]] = result[1]
#        print(len(POS_DICTIONARY))

#    for d_words in D_words:
#        if d_words in POS_DICTIONARY:
#            continue
#        else:
#            pool.apply_async(GetPOSTagParallel, args=(d_words), callback=CollectResults)

#    pool.close()
#    pool.join()

#    return

def WritePOSToExcel(path):
    with xlsxwriter.Workbook(path, {'strings_to_urls': False}) as workbook:
        worksheet = workbook.add_worksheet();
        row = 0;
        col = 0;
        worksheet.write(row, col, 'Text'); col += 1;
        worksheet.write(row, col, 'POS'); col += 1;
        row += 1;
        for key in POS_DICTIONARY:
            col = 0
            worksheet.write(row, col, key); col += 1;
            worksheet.write(row, col, ' '.join(POS_DICTIONARY[key])); col += 1;
            row += 1;
    return

def LoadPOSExcel(path):
    if os.path.isfile(path):
        df = pd.read_excel(path, usecols=range(0, 50))
        for i in range(len(df['Text'])):
            text = str(df['Text'][i])
            pos = str(df['POS'][i])
            POS_DICTIONARY[text] = [str(x) for x in pos.split(' ')]
    return

def GetTextStatInfo(text):
    le_c = textstat.lexicon_count(text, removepunct=True)
    ts = textstat.text_standard(text, float_output=True)

    return le_c, ts

def TextTokenizer(text):
    new_text = text
    new_text = new_text.replace('/', ' ')
    new_text = new_text.replace('.', ' ')
    new_text = new_text.replace('-', ' ')
    new_text = new_text.replace('_', ' ')
    new_text = new_text.replace('%', ' ')
    new_text = new_text.replace(',', '')
    new_text = new_text.replace('(', '')
    new_text = new_text.replace(')', '')
    new_text = new_text.replace('!', '')
    new_text = new_text.replace(';', '')
    new_text = new_text.replace('$', '')
    new_text = new_text.replace('"', '')
    new_text = new_text.replace('?', '')
    new_text = new_text.replace(':', '')
    new_text = new_text.replace('~', '')
    new_text = new_text.replace('*', '')
    new_text = new_text.replace(" '", '')
    new_text = new_text.replace("' ", '')

    #tokens = nltk.WhitespaceTokenizer().tokenize(new_text)
    tokens = nltk.word_tokenize(new_text)

    #for i in range(len(tokens)):
    #    tokens[i] = tokens[i].replace('.', '')
    #    tokens[i] = tokens[i].replace(',', '')
    #    tokens[i] = tokens[i].replace(')', '')
    #    tokens[i] = tokens[i].replace('(', '')
    #    tokens[i] = tokens[i].replace('!', '')
    #    tokens[i] = tokens[i].replace(';', '')
    #    tokens[i] = tokens[i].replace('$', '')

    return tokens

def CleanText(text):
    clean_text = text
    clean_text = clean_text.replace('&nbsp', ' ')
    clean_text = clean_text.replace('…', '...')
    clean_text = clean_text.replace('‘', "'")
    clean_text = clean_text.replace('’', "'")
    clean_text = clean_text.replace('—', '-')
    clean_text = clean_text.replace('–', '-')
    clean_text = clean_text.replace('“', '"')
    clean_text = clean_text.replace('”', '"')
    clean_text = clean_text.replace('£', '$')
    clean_text = clean_text.replace('`', "'")
    clean_text = clean_text.replace('`', "'")
    clean_text = clean_text.replace('\x92', '')
    clean_text = clean_text.replace('\x93', '')
    clean_text = clean_text.replace('\x94', '')

    clean_text = re.sub(r"http\S+", "", clean_text)

    clean_text = ' '.join(clean_text.split())

    return clean_text

def GetNumberOfIncorrectSpelling(text):
    tokens = TextTokenizer(text)

    misspelled = SPELL.unknown(tokens)

    #for i in range(len(misspelled)):
    #    misspelled[i] = ''.join(e for e in misspelled[i] if e.isalnum())

    misspelled.discard('')

    random_val = random.randint(0, 100)
    if random_val > 99:
        print(misspelled)
    return len(misspelled)

if __name__ == '__main__':
    SplitData()