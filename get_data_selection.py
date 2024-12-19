# create dictionary where each doc gets an identifier and is linked to its name, its year, its annotator, the half century correspondent to the  year
import os
import random

def get_filepath_list(root_path):
    """
    Get complete filepaths leading to all relevant training data documents in a list
    :param root_path: str
    """
    file_list = []
    for root, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_list.append(os.path.join(root, filename))
    return(file_list)

def create_data_inventory(file_list):
    """
    creates a data inventory according to filepath and filename structure of data
    """
    data_inventory = []
    for x in file_list:
        d = dict()
        spl = x.split('/')
        if spl[3] == 'train_2':
            round = '2'
        if spl[3] == 'train_3':
            round = '3'
        if spl[3] == 'train_4':
            round = '4'
        if spl[4] == 'special_topic_ESTA':
            round = '3_ESTA'
        year = int((spl[-1].split(' -')[1].strip()))
        if int((spl[-1].split(' -')[1].strip())) <1650:
            centuryhalf = '1600-1650'
        if int((spl[-1].split(' -')[1].strip())) in range (1650, 1700):
            centuryhalf = '1650-1700'
        if int((spl[-1].split(' -')[1].strip())) in range(1700, 1750):
            centuryhalf = '1700-1750'
        if int((spl[-1].split(' -')[1].strip())) in range(1750, 1800):
            centuryhalf = '1750-1800'
        if round == '2':
            inv_nr = ((spl[-1].split('_'))[2]).split(' - ')[0]
            scan_nrs = ((spl[-1].split('_'))[3]).split(' - ')[0]
        if round == '3' or round == '3_ESTA':
            inv_nr = ((spl[-1].split('_'))[4]).split(' - ')[0]
            scan_nrs = ((spl[-1].split('_'))[5]).split(' - ')[0]
        if round == '4':
            inv_nr = ((spl[-1].split('_'))[2]).split(' - ')[0]
            scan_nrs = ((spl[-1].split('_'))[3]).split(' - ')[0]
        d['original_filename'] = spl[-1]
        d['round'] = round
        d['inv_nr'] = inv_nr
        d['scan_nrs'] = scan_nrs
        d['year'] = year
        d['century_half'] = centuryhalf
        data_inventory.append(d)
    return(data_inventory)


def file_selection(data_inventory):
    """
    makes a certain selection of the data as validation set depending on paramaters
    """
    parameter = input('On which parameter would you like to select your testdata? Type "century", "annotation round" or "ESTA"')
    percentage = '100'
    testfiles=[]
    if parameter == 'century':
        centuryhalf = input('Which century half would you like to use as testdata?')
        for d in data_inventory:
            if d['century_half'] == centuryhalf:
                testfiles.append(d['original_filename'])
        doc_or_percentage = input('Would you like to use all of the ' + str(
            len(testfiles)) + ' documents, would you like to randomly select one of them or would you like to select a percentage of each document? Type "all", "doc" or "percentage"')
        if doc_or_percentage == 'doc':
            testfiles = random.choice(testfiles)
        if doc_or_percentage == 'percentage':
            percentage = input('You have selected ' + centuryhalf +'. For this round, there are ' + str(
                        len(testfiles)) + ' files available. What percentage of each doc would you like for your test data?')
    if parameter == 'ESTA':
        for d in data_inventory:
            if d['round'] == '3_ESTA':
                testfiles.append(d['original_filename'])
        doc_or_percentage = input('Would you like to use all of the ' + str(
            len(testfiles)) + ' documents, would you like to randomly select one of them or would you like to select a percentage of each document? Type "all", "doc" or "percentage"')
        if doc_or_percentage == 'doc':
            testfiles = random.choice(testfiles)
        if doc_or_percentage == 'percentage':
            percentage = input('For the ESTA data, there are ' + str(
                len(testfiles)) + ' files available. What percentage of each doc would you like for your test data?')
    if parameter == 'annotation round':
        round = input('From which round would you like to select test data? Type 2, 3 or 4.')
        if round == '3':
            for d in data_inventory:
                if d['round'] == '3':
                    testfiles.append(d['original_filename'])
            doc_or_percentage = input('Would you like to use all of the ' + str(
                len(testfiles)) + ' documents, would you like to randomly select one of them or would you like to select a percentage of each document? Type "all", "doc" or "percentage"')
            if doc_or_percentage == 'doc':
                testfiles = random.choice(testfiles)
            if doc_or_percentage == 'percentage':
                percentage = input('You have selected round 3. For this round, there are ' + str(
                    len(testfiles)) + ' files available. What percentage of each doc would you like for your test data?')
        if round == '2':
            for d in data_inventory:
                if d['round'] == '2':
                    testfiles.append(d['original_filename'])
            doc_or_percentage = input('Would you like to use all of the ' + str(
                len(testfiles)) + ' documents, would you like to randomly select one of them or would you like to select a percentage of each document? Type "all", "doc" or "percentage"')
            if doc_or_percentage == 'doc':
                testfiles = random.choice(testfiles)
            if doc_or_percentage == 'percentage':
                percentage = input('You have selected round 2. For this round, there are ' + str(
                    len(testfiles)) + ' files available. What percentage of each doc would you like for your test data?')
        if round == '4':
            for d in data_inventory:
                if d['round'] == '4':
                    testfiles.append(d['original_filename'])
            doc_or_percentage = input('Would you like to use all of the ' + str(
                len(testfiles)) + ' documents, would you like to randomly select one of them or would you like to select a percentage of each document? Type "all", "doc" or "percentage"')
            if doc_or_percentage == 'doc':
                testfiles = random.choice(testfiles)
            if doc_or_percentage == 'percentage':
                percentage = input('You have selected round 3. For this round, there are ' + str(
                    len(testfiles)) + ' files available. What percentage of each doc would you like for your test data?')
    print('Your selection has been made and the data settings has been saved as metadata')
    settings = dict()
    settings['parameter'] = parameter
    settings['percentage'] = percentage
    settings['filenames'] = testfiles
    metadata = []
    if type(testfiles) == list:
        for filename in testfiles:
            for d in data_inventory:
                if d['original_filename'] == filename:
                    metadata.append(d)
    if type(testfiles) == str:
        for d in data_inventory:
            if d['original_filename'] == testfiles:
                metadata.append(d)
    settings['metadata'] = metadata
    return(settings)

def file_selection_invnr(root_path, inv_nr):
    """
    Single test file selection on basis of inventory number
    Creates "settings.json" and adds metadata of test file
    """

    filepaths = get_filepath_list(root_path)

    settings = dict()
    metadata = []

    data_inventory = create_data_inventory(filepaths)
    for d in data_inventory:
        if d['inv_nr'] == inv_nr:
            metadata = d
    settings['metadata_testfile'] = metadata
    return (settings)

def create_mixed_validationset():
    """
    Create a validation set with x amount of chunks from each different document
    """

if __name__ == "__main__":
    root_path = "data/json_per_doc/"
    filepaths=get_filepath_list(root_path)
    data_inv=create_data_inventory(filepaths)
    settings = file_selection(data_inv)
    #print(settings)

