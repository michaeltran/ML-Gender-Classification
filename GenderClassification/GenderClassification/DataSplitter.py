import nltk
import argparse
import xlsxwriter
import pandas as pd
from random import shuffle

def SplitData():
    ## Get Command-line Arguments #################
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='data/blog-gender-dataset.xlsx', help='')
    opts = parser.parse_args()
    ###############################################

    ## Build the Training Set and Testing Set #####
    names = ['Text', 'Classification']
    df = pd.read_excel(opts.data, header=None, names=names, usecols="A,B")

    data_male_text = []
    data_female_text = []

    training_data_text = []
    training_data_classification = []
    testing_data_text = []
    testing_data_classification = []

    # Prepare and Sanitize data - 0 = Female, 1 = Male
    for i in range(len(df['Text'])):
        text = df['Text'][i]
        classification = df['Classification'][i]
        if text == text and classification == classification:
            #tokens = nltk.word_tokenize(text)
            #tagged = nltk.pos_tag(tokens)
            #pos_text = []

            #for (a,b) in tagged:
            #    #pos_text.append('%s_%s' % (a, b))
            #    pos_text.append('%s' % (a))

            classification = classification.strip().upper()
            if classification == 'M':
                #data_male_text.append(' '.join(pos_text))
                data_male_text.append(text)
            elif classification == 'F':
                #data_female_text.append(' '.join(pos_text))
                data_female_text.append(text)
            else:
                print('Classification Error: %s is not defined.' % (classification))
                return

    shuffle(data_male_text)
    shuffle(data_female_text)
    # Split out training dataset
    len_data = 0
    len_data = min(int(len(data_male_text)), int(len(data_female_text)))
    len_training_data = int(len_data * (9/10) * 2)
    for i in range(len_training_data):
        if i % 2 == 1:
            # Male
            training_data_text.append(data_male_text[0])
            training_data_classification.append(1)
            del data_male_text[0]
        else:
            # Female
            training_data_text.append(data_female_text[0])
            training_data_classification.append(0)
            del data_female_text[0]

    # Split out testing dataset
    for i in range(min(len(data_male_text), len(data_female_text))):
        testing_data_text.append(data_male_text[i])
        testing_data_classification.append(1)
        testing_data_text.append(data_female_text[i])
        testing_data_classification.append(0)

    # Save Training Data
    workbook = xlsxwriter.Workbook('data/train_data.xlsx') 
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    worksheet.write(row, col, 'Classification')
    worksheet.write(row, col + 1, 'Text')
    worksheet.write(row, col + 2, 'Length')
    worksheet.write(row, col + 3, 'F-Measure')
    worksheet.write(row, col + 4, 'GPF1')
    worksheet.write(row, col + 5, 'GPF2')
    worksheet.write(row, col + 6, 'GPF3')
    worksheet.write(row, col + 7, 'GPF4')
    worksheet.write(row, col + 8, 'GPF5')
    worksheet.write(row, col + 9, 'GPF6')
    worksheet.write(row, col + 10, 'GPF7')
    worksheet.write(row, col + 11, 'GPF8')
    worksheet.write(row, col + 12, 'GPF9')
    worksheet.write(row, col + 13, 'GPF10')
    worksheet.write(row, col + 14, 'FA1')
    worksheet.write(row, col + 15, 'FA2')
    worksheet.write(row, col + 16, 'FA3')
    worksheet.write(row, col + 17, 'FA4')
    worksheet.write(row, col + 18, 'FA5')
    worksheet.write(row, col + 19, 'FA6')
    worksheet.write(row, col + 20, 'FA7')
    worksheet.write(row, col + 21, 'FA8')
    worksheet.write(row, col + 22, 'FA9')
    worksheet.write(row, col + 23, 'FA10')
    worksheet.write(row, col + 24, 'FA11')
    worksheet.write(row, col + 25, 'FA12')
    worksheet.write(row, col + 26, 'FA13')
    worksheet.write(row, col + 27, 'FA14')
    worksheet.write(row, col + 28, 'FA15')
    worksheet.write(row, col + 29, 'FA16')
    worksheet.write(row, col + 30, 'FA17')
    worksheet.write(row, col + 31, 'FA18')
    worksheet.write(row, col + 32, 'FA19')
    worksheet.write(row, col + 33, 'FA20')
    worksheet.write(row, col + 34, 'FA21')
    worksheet.write(row, col + 35, 'FA22')
    worksheet.write(row, col + 36, 'FA23')
    row += 1
    for i in range(len(training_data_text)):
        text = training_data_text[i]
        gpf = GetGenderPreferentialFeatures(text)
        fa = GetFactorAnalysis(text)
        worksheet.write(row, col, training_data_classification[i])
        worksheet.write(row, col + 1, text)
        worksheet.write(row, col + 2, len(text))
        worksheet.write(row, col + 3, GetFMeasure(text))
        worksheet.write(row, col + 4, gpf[0])
        worksheet.write(row, col + 5, gpf[1])
        worksheet.write(row, col + 6, gpf[2])
        worksheet.write(row, col + 7, gpf[3])
        worksheet.write(row, col + 8, gpf[4])
        worksheet.write(row, col + 9, gpf[5])
        worksheet.write(row, col + 10, gpf[6])
        worksheet.write(row, col + 11, gpf[7])
        worksheet.write(row, col + 12, gpf[8])
        worksheet.write(row, col + 13, gpf[9])
        worksheet.write(row, col + 14, fa[0])
        worksheet.write(row, col + 15, fa[1])
        worksheet.write(row, col + 16, fa[2])
        worksheet.write(row, col + 17, fa[3])
        worksheet.write(row, col + 18, fa[4])
        worksheet.write(row, col + 19, fa[5])
        worksheet.write(row, col + 20, fa[6])
        worksheet.write(row, col + 21, fa[7])
        worksheet.write(row, col + 22, fa[8])
        worksheet.write(row, col + 23, fa[9])
        worksheet.write(row, col + 24, fa[10])
        worksheet.write(row, col + 25, fa[11])
        worksheet.write(row, col + 26, fa[12])
        worksheet.write(row, col + 27, fa[13])
        worksheet.write(row, col + 28, fa[14])
        worksheet.write(row, col + 29, fa[15])
        worksheet.write(row, col + 30, fa[16])
        worksheet.write(row, col + 31, fa[17])
        worksheet.write(row, col + 32, fa[18])
        worksheet.write(row, col + 33, fa[19])
        worksheet.write(row, col + 34, fa[20])
        worksheet.write(row, col + 35, fa[21])
        worksheet.write(row, col + 36, fa[22])
        row += 1
    workbook.close()

    # Save Testing Data
    workbook = xlsxwriter.Workbook('data/test_data.xlsx') 
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    worksheet.write(row, col, 'Classification')
    worksheet.write(row, col + 1, 'Text')
    worksheet.write(row, col + 2, 'Length')
    worksheet.write(row, col + 3, 'F-Measure')
    worksheet.write(row, col + 4, 'GPF1')
    worksheet.write(row, col + 5, 'GPF2')
    worksheet.write(row, col + 6, 'GPF3')
    worksheet.write(row, col + 7, 'GPF4')
    worksheet.write(row, col + 8, 'GPF5')
    worksheet.write(row, col + 9, 'GPF6')
    worksheet.write(row, col + 10, 'GPF7')
    worksheet.write(row, col + 11, 'GPF8')
    worksheet.write(row, col + 12, 'GPF9')
    worksheet.write(row, col + 13, 'GPF10')
    worksheet.write(row, col + 14, 'FA1')
    worksheet.write(row, col + 15, 'FA2')
    worksheet.write(row, col + 16, 'FA3')
    worksheet.write(row, col + 17, 'FA4')
    worksheet.write(row, col + 18, 'FA5')
    worksheet.write(row, col + 19, 'FA6')
    worksheet.write(row, col + 20, 'FA7')
    worksheet.write(row, col + 21, 'FA8')
    worksheet.write(row, col + 22, 'FA9')
    worksheet.write(row, col + 23, 'FA10')
    worksheet.write(row, col + 24, 'FA11')
    worksheet.write(row, col + 25, 'FA12')
    worksheet.write(row, col + 26, 'FA13')
    worksheet.write(row, col + 27, 'FA14')
    worksheet.write(row, col + 28, 'FA15')
    worksheet.write(row, col + 29, 'FA16')
    worksheet.write(row, col + 30, 'FA17')
    worksheet.write(row, col + 31, 'FA18')
    worksheet.write(row, col + 32, 'FA19')
    worksheet.write(row, col + 33, 'FA20')
    worksheet.write(row, col + 34, 'FA21')
    worksheet.write(row, col + 35, 'FA22')
    worksheet.write(row, col + 36, 'FA23')
    row += 1
    for i in range(len(testing_data_text)):
        text = testing_data_text[i]
        gpf = GetGenderPreferentialFeatures(text)
        fa = GetFactorAnalysis(text)
        worksheet.write(row, col, testing_data_classification[i])
        worksheet.write(row, col + 1, text)
        worksheet.write(row, col + 2, len(text))
        worksheet.write(row, col + 3, GetFMeasure(text))
        worksheet.write(row, col + 4, gpf[0])
        worksheet.write(row, col + 5, gpf[1])
        worksheet.write(row, col + 6, gpf[2])
        worksheet.write(row, col + 7, gpf[3])
        worksheet.write(row, col + 8, gpf[4])
        worksheet.write(row, col + 9, gpf[5])
        worksheet.write(row, col + 10, gpf[6])
        worksheet.write(row, col + 11, gpf[7])
        worksheet.write(row, col + 12, gpf[8])
        worksheet.write(row, col + 13, gpf[9])
        worksheet.write(row, col + 14, fa[0])
        worksheet.write(row, col + 15, fa[1])
        worksheet.write(row, col + 16, fa[2])
        worksheet.write(row, col + 17, fa[3])
        worksheet.write(row, col + 18, fa[4])
        worksheet.write(row, col + 19, fa[5])
        worksheet.write(row, col + 20, fa[6])
        worksheet.write(row, col + 21, fa[7])
        worksheet.write(row, col + 22, fa[8])
        worksheet.write(row, col + 23, fa[9])
        worksheet.write(row, col + 24, fa[10])
        worksheet.write(row, col + 25, fa[11])
        worksheet.write(row, col + 26, fa[12])
        worksheet.write(row, col + 27, fa[13])
        worksheet.write(row, col + 28, fa[14])
        worksheet.write(row, col + 29, fa[15])
        worksheet.write(row, col + 30, fa[16])
        worksheet.write(row, col + 31, fa[17])
        worksheet.write(row, col + 32, fa[18])
        worksheet.write(row, col + 33, fa[19])
        worksheet.write(row, col + 34, fa[20])
        worksheet.write(row, col + 35, fa[21])
        worksheet.write(row, col + 36, fa[22])
        row += 1
    workbook.close()
    ###############################################
    print("Done")

def GetFMeasure(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

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
        pos = tagged[i][1]
        if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
            freq['noun'] += 1
        elif pos in ['JJ', 'JJR', 'JJS']:
            freq['adj'] += 1
        elif pos in ['IN']:
            freq['prep'] += 1
        elif pos in ['DET']:
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
    for word in text.split():
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
                      'inadequate', 'poor', 'paltry', 'trifling', 'cheap', 'mean', 'shabby', 'scrubby', 'stunted', 'unimportant', 'beggarly', 'insignificant', 'dismal', 'pitiful', 'worthless', 
                      'despicable', 'sad', 'greived', 'mournful', 'melancholy']:
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
    for word in text.split():
        word = word.lower()
        for i in range(len(words_in_factor)):
            if word in words_in_factor[i]:
                f[i] += 1
    return f

SplitData()