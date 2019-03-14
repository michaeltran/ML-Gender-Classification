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
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            pos_text = []

            for (a,b) in tagged:
                pos_text.append('%s_%s' % (a, b))
                #pos_text.append('%s' % (b))

            classification = classification.strip().upper()
            if classification == 'M':
                data_male_text.append(' '.join(pos_text))
            elif classification == 'F':
                data_female_text.append(' '.join(pos_text))
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
    for i in range(len(training_data_text)):
        worksheet.write(row, col, training_data_classification[i])
        worksheet.write(row, col + 1, training_data_text[i])
        row += 1
    workbook.close()

    # Save Testing Data
    workbook = xlsxwriter.Workbook('data/test_data.xlsx') 
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for i in range(len(testing_data_text)):
        worksheet.write(row, col, testing_data_classification[i])
        worksheet.write(row, col + 1, testing_data_text[i])
        row += 1
    workbook.close()
    ###############################################


SplitData()