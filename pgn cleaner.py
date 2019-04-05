import numpy as np
import os
from random import randint
from pathlib import Path
import time
from selenium.webdriver.common.action_chains import ActionChains
# import selenium.webdriver as webdriver
import selenium.webdriver.support.ui as ui
import pickle
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException, ElementNotInteractableException
import hashlib

curdir = os.getcwd()

def flatten_to_board_arr(board_arr):
    shell = np.zeros((8, 8), dtype='U10')
    piece_list = ['bRo', 'bKn', 'bBi', 'bQu', 'bKi', 'bPa',
                  'wPa', 'wKi', 'wQu', 'wBi', 'wKn', 'wRo']

    positions = np.asarray(board_arr[:768]).reshape((8, 8, 12))
    score = np.asarray(board_arr[768])

    for row in range(8):
        for column in range(8):
            try:
                this_piece = piece_list[list(positions[row, column, :]).index(1)]
            except ValueError:
                this_piece = '---'

            shell[row, column] = this_piece

    return shell


def count_down():
    print('3')
    time.sleep(1)
    print('2')
    time.sleep(1)
    print('1')
    time.sleep(1)
    return


def one_hot_flatten(board_arr):
    '''
    :param board_arr: an 8x8 array presentation of the chess board
    :return: a flattened, one-hot-encoded array of shape (12x8x8)
    '''
    piece_list = ['bRo', 'bKn', 'bBi', 'bQu', 'bKi', 'bPa',
                  'wPa', 'wKi', 'wQu', 'wBi', 'wKn', 'wRo']

    board_one_hot_flat = []
    for row in range(8):
        for column in range(8):
            # print(board_arr[row,column])
            if board_arr[row, column] == '---':
                one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            else:
                one_hot = [int(1) if piece_list.index(board_arr[row][column]) == x else int(0) for x in range(12)]
            board_one_hot_flat = board_one_hot_flat + one_hot

    return np.asarray(board_one_hot_flat)


def pgn_sample(pgn_file, nr_samples_per_game, min_rel_sample_length, max_rel_sample_length):

    with open(pgn_file) as fp:
        pgn_list = []
        # seperate the game notation from the game metadata
        pgns = [element.split('\n\n')[1] for element in fp.read().split(sep='\n\n[') if element[0] != '['][1500:]

        for pgn in pgns:
            pgn = pgn.replace('\n', ' ')

            moves = pgn.split('.')
            nr_moves = len(moves) - 1

            if nr_moves < 5:
                continue

            for i in range(nr_samples_per_game):
                min_moves = (round(nr_moves/100 * min_rel_sample_length) -1)
                max_moves = (round(nr_moves/100 * max_rel_sample_length) + 1)
                if max_moves == nr_moves:
                    max_moves -= 1

                # print('nr_moves = {}\nmin_moves = {}\nmax_moves = {}\n\n'.format(nr_moves, min_moves, max_moves))
                n = randint(min_moves, max_moves)

                sample_pgn = '.'.join(moves[:(n + 1)])[:-2].strip()
                sample_pgn = sample_pgn.split(' ')

                # flip a coin to decide if we take the pgn with white or black to move
                heads = randint(0, 1)
                if heads:
                    sample_pgn = ' '.join(sample_pgn[:-1])
                    # print('black to move: ', sample_pgn)
                    white_to_move = 0
                else:
                    sample_pgn = ' '.join(sample_pgn)
                    # print('white to move: ', sample_pgn)
                    white_to_move = 1

                pgn_list.append((sample_pgn, white_to_move))

    return pgn_list


def get_pgn_score(pgn_list):
    train_data_batch = []
    md5 = hashlib.md5()
    count = 0

    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options

    os.chdir(curdir)
    driver_dir = str(Path.cwd() / 'geckodriver.exe')
    driver = webdriver.Firefox(executable_path=driver_dir)
    driver.set_page_load_timeout(1800)  # set time_out time
    driver.get('http://lichess.org/analysis')
    time.sleep(2)
    driver.maximize_window()
    time.sleep(1)
    # turn on engine
    wait = ui.WebDriverWait(driver, 10)
    wait.until(lambda driver: driver.find_element_by_class_name('switch'))
    driver.find_element_by_class_name('switch').click()
    time.sleep(0.5)

    for pgn in pgn_list:
        try:
            count += 1

            # INITIALIZE VARIABLES
            white_to_move = pgn[1]
            pgn = pgn[0]

            # fill in the pgn
            wait.until(lambda driver: driver.find_element_by_class_name('pgn'))
            driver.find_element_by_class_name('pgn').find_elements_by_css_selector('*')[1].clear()
            time.sleep(1/randint(3,10))
            driver.find_element_by_class_name('pgn').find_elements_by_css_selector('*')[1].send_keys(pgn)

            time.sleep(3/randint(3, 10))
            # hover over pgn field so that submit button is revealed
            hov = ActionChains(driver).move_to_element(driver.find_element_by_class_name('pgn'))
            hov.perform()
            hov.perform()
            # Submit pgn (but first wait for the button to appear)
            wait.until(lambda driver: driver.find_element_by_class_name('action').text == 'PGN importeren')
            while True:
                try:
                    hov.perform()
                    driver.find_element_by_class_name('pgn').find_element_by_class_name('action').find_elements_by_css_selector('*')[0].click()
                    break
                except ElementNotInteractableException:
                    continue
            # fetch the score
            while True:
                try:
                    wait.until(lambda driver: driver.find_element_by_class_name('info').text.split(' ')[0] != 'Laden')
                    wait.until(lambda driver: driver.find_element_by_class_name('info').text.split(' ')[1][1] != '/')
                    wait.until(lambda driver: int(driver.find_element_by_class_name('info').text.split(' ')[1][:2]) >= 18)
                    break
                except ValueError:
                    continue
            score = driver.find_element_by_class_name('lichess_ground').find_elements_by_css_selector('*')[0].text.split('\n')[0]
            time.sleep(1 / randint(3, 10))

            # ARRAY FOR BOARD POSITION
            board_arr = np.zeros((8, 8), dtype='<U10')

            while True:
                try:
                    n_pieces = len(driver.find_element_by_class_name('cg-board').find_elements_by_css_selector('*')[2:])
                    break
                except:
                    continue

            for i in range(n_pieces):
                while True:
                    try:
                        name = driver.find_element_by_class_name('cg-board').find_elements_by_css_selector('*')[2:][i].get_attribute('class')
                        location = driver.find_element_by_class_name('cg-board').find_elements_by_css_selector('*')[2:][i].location
                        break
                    except StaleElementReferenceException:
                        continue


                if name[:5] != 'black' and name[:5] != 'white':
                    continue

                row_index = int((location['y'] - 59) / 64)
                column_index = int((location['x'] - 668) / 64)
                # print(name, location)
                # print('array position :', str(row_index) + ',' + str(column_index))
                # print()
                board_arr[row_index, column_index] = str(name.split(' ')[0][0] + name.split(' ')[1][0].upper() + name.split(' ')[1][1])

            # fill the empty squares with '---'
            for x in range(8):
                for y in range(8):
                    if board_arr[x,y] == '':
                        board_arr[x,y] = '---'
            one_hot_board = str(one_hot_flatten(board_arr)).replace('\n', '')[1:-1]
            md5.update(one_hot_board.encode('utf-8'))
            this_hash = md5.hexdigest()

            row = [this_hash, (one_hot_board + ' ' + str(white_to_move)), score]
            train_data_batch.append(row)

            # UPDATE DATASET EVERY 250 EXAMPLES
            if count % 250 == 0:
                rows_to_append = []
                count = 0
                with open(curdir + '/data/data.pickle', 'rb') as pickle_in:
                    train_data = np.asarray(pickle.load(pickle_in))
                    nr_rows = train_data.shape[0]
                    hashes = train_data[:,0]
                for row in train_data_batch:
                    if row[0] not in hashes:
                        # row = np.asarray(row).reshape(1, 2)
                        # print('row type = ', type(row))
                        # print('row = ', row)
                        rows_to_append.append(row)
                        # print('rows to append = ', rows_to_append)
                        continue
                    else:
                        print('position already in datset')
                        continue

                rows_to_append = np.asarray(rows_to_append).reshape(len(rows_to_append),len(rows_to_append[0]))
                new_data = np.concatenate((train_data, rows_to_append))

                with open("data.pickle","wb") as pickle_out:
                    pickle.dump(new_data, pickle_out)
                    print('added {} more rows to dataset'.format(new_data.shape[0] - nr_rows))
                    print()
                train_data_batch = []
                continue

        except (TimeoutException, IndexError) as e:
            continue

    driver.close()
    driver.quit()
    return train_data_batch

#
# dataset = get_pgn_score(pgn)
#
# with open("vb.pickle","wb") as pickle_out:
#     pickle.dump(dataset, pickle_out)

def get_train_data(path_to_pickle):
    pgn_dir = curdir + '/pgn/Players/'
    for file in os.listdir(pgn_dir):
        if '.pgn' in file:
            print('---------------------------------------------------------------------------------------------')
            print(file)
            print('---------------------------------------------------------------------------------------------')
            pgn_list = pgn_sample((pgn_dir + file), 5, 50, 90)
            get_pgn_score(pgn_list)
            print('file processed, moving on to the next...')
            print()
    return

get_train_data(curdir + '/data/data.pickle')

