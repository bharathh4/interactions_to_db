import dbhelper
import orm_helper
import os
import re
import math
from constants import DATA_DIR, DB_PATH, DATA_SOURCE
from collections import Counter
import calendar
from datetime import date

from tabulate import tabulate

try:
    import numpy as np
    import matplotlib.pyplot as plt
except:
    print 'Could not import numpy. Need numpy for some methods. Perhaps some other import error happend'


if DATA_SOURCE is 'sqlite':

    def mean(store='AUS', param='conf'):
        Transcriptions = orm_helper.Transcriptions
        confs = [getattr(row, param) for row in Transcriptions.select().where(
            Transcriptions.store == store)]
        return np.mean(np.array(confs))

    def variance(store='AUS', param='conf'):
        Transcriptions = orm_helper.Transcriptions
        confs = [getattr(row, param) for row in Transcriptions.select().where(
            Transcriptions.store == store)]
        return np.var(confs)

    def histogram(store='AUS', param='conf'):
        Transcriptions = orm_helper.Transcriptions
        confs = [getattr(row, param) for row in Transcriptions.select().where(
            Transcriptions.store == store)]
        plt.hist(confs, 100)
        plt.show()

    def get_sample_row():
        Transcriptions = orm_helper.Transcriptions
        rows = [row for row in Transcriptions.select()]
        return rows[0]

    def get_row(params):
        for row in Transcriptions.select().where(Transcriptions.store == store):
            print row

    def main():
        print 'The data source has been set to sqlite'
        # print np.sqrt(get_conf_variance(storename='AUS'))
        # conf_histogram()
        pass

elif DATA_SOURCE is 'csv':

    import csv

    def hasNumbers(inputString):
        '''Returns true if numbers found in string'''
        return any(char.isdigit() for char in inputString)

    def interaction2csv(interactionfilename, output_csvfilename):
        '''Converts a interaction file to a csv file and saves it in a default 
        location for later retrieval'''
        with open(interactionfilename, 'r') as fread:
            str_to_write = ''.join([line for line in fread if len(line) > 100])
            with open(output_csvfilename, 'w') as foutput:
                foutput.write(str_to_write)

    def get_reader(filename):
        '''Gets the reader object for a csv/interaction file'''

        if (filename.endswith('interaction') or
                filename.endswith('Interaction') or
                filename.endswith('interactions') or
                filename.endswith('Interactions')):

            # convert interaction file to csv
            output_csvfilename = os.path.join(DATA_DIR, 'converted.csv')
            interactionfilename = filename
            interaction2csv(interactionfilename, output_csvfilename)
            csv_filename = output_csvfilename
        else:
            csv_filename = filename
        try:
            output_collated_csv_file = open(csv_filename, 'r')
            reader = csv.reader(output_collated_csv_file,
                                delimiter=',', quotechar='|')
            return reader
        except:
            print 'DEBUG : get_reader() The file couldnt be opened'

    def process(row):
        '''The lumenvox interaction file has a certain format. The function 
        unpacks every interaction file row and extracts information'''
        (transcript_si,
         transcript,
         decode_si,
         decode,
         conf,
         decode_time,
         callsrepath,
         _,
         _,
         _,
         acoustic_model,
         _,
         _,
         _,
         _,
         _,
         _,
         _,
         _
         ) = row

        (date, time, milliseconds, grammarlevel, fullname, oration_id,
         chain, store) = callsrepath.split('\\')[-1].split('_')

        try:
            firstname, lastname = fullname.split(' ')
        except:
            firstname, lastname = fullname, fullname

        return (transcript_si, transcript, decode_si, decode, int(conf),
                decode_time, callsrepath, acoustic_model, date, time,
                milliseconds, grammarlevel, firstname, lastname,
                oration_id, chain, store.replace('.callsre', ''))

    def compute_ca(transcript, decode, conf, threshold):
        '''Computes correct accept rate. Takes a transcript semantic intent 
        and decode semantic intent, the decode confidence and threshold as 
        parameters'''
        if transcript == decode and int(conf) >= threshold:
            return True
        else:
            return False

    def compute_fr(transcript, decode, conf, threshold):
        '''Computes false reject rate. Takes a transcript semantic intent and 
        decode semantic intent, the decode confidence and threshold as 
        parameters'''
        if transcript == decode and int(conf) < threshold:
            return True
        else:
            return False

    def compute_fa(transcript, decode, conf, threshold):
        '''Computes false accept rate. Takes a transcript semantic intent and 
        decode semantic intent, the decode confidence and threshold as 
        parameters'''
        if transcript != decode and int(conf) >= threshold:
            return True
        else:
            return False

    def compute_cr(transcript, decode, conf, threshold):
        '''Computes correct reject rate. Takes a transcript semantic intent and
        decode semantic intent, the decode confidence and threshold as 
        parameters'''
        if transcript != decode and int(conf) < threshold:
            return True
        else:
            return False

    def get_overall_metrics(filename, threshold=100, day=None):
        '''computes CA, CR, FR, FA overall and also for in-grammar and out-grammar.
        Returns dictionary of overall CA, CR, FR, FA, in grammar CA, CR, FR, FA
        and out of grammar CA, CR, FR, FA'''
        ca_list, cr_list, fa_list, fr_list = [], [], [], []

        reader = get_reader(filename)
        for row in reader:
            (transcript_si, transcript, decode_si, decode, conf, decode_time,
             callsrepath, acoustic_model,
             date, time, milliseconds, grammarlevel, firstname, lastname, oration_id,
             chain, store) = process(row)
             
            # 
            if day is not None: 
                expr = ur'(2016|2017|2018|2019)([0-1][0-9])([0-3][0-9])'
                p = re.compile(expr)
                result = re.search(p, date)
                Year, Month, Day = map(int, result.groups()[:3])
                if Day is not day:
                    continue

            ca, fa, cr, fr = (compute_ca(transcript_si, decode_si, conf, threshold),
                              compute_fa(transcript_si, decode_si,
                                         conf, threshold),
                              compute_cr(transcript_si, decode_si,
                                         conf, threshold),
                              compute_fr(transcript_si, decode_si, conf, threshold))

            if transcript_si in ['~No interpretations']:
                gram_status = 'OOG'
            else:
                gram_status = 'ING'

            ca_list.append((ca, gram_status))
            fa_list.append((fa, gram_status))
            cr_list.append((cr, gram_status))
            fr_list.append((fr, gram_status))

        ca_rate = 100 * \
            sum([val for val, gram_status in ca_list]) / float(len(ca_list))
        fa_rate = 100 * \
            sum([val for val, gram_status in fa_list]) / float(len(fa_list))
        cr_rate = 100 * \
            sum([val for val, gram_status in cr_list]) / float(len(cr_list))
        fr_rate = 100 * \
            sum([val for val, gram_status in fr_list]) / float(len(fr_list))

        ingram_ca_list = [val for val,
                          gram_status in ca_list if gram_status == 'ING']
        ingram_fa_list = [val for val,
                          gram_status in fa_list if gram_status == 'ING']
        ingram_fr_list = [val for val,
                          gram_status in fr_list if gram_status == 'ING']
        ingram_cr_list = [val for val,
                          gram_status in cr_list if gram_status == 'ING']

        ingram_ca_rate = 100 * \
            sum([val for val in ingram_ca_list]) / float(len(ingram_ca_list))
        ingram_fa_rate = 100 * \
            sum([val for val in ingram_fa_list]) / float(len(ingram_fa_list))
        ingram_fr_rate = 100 * \
            sum([val for val in ingram_fr_list]) / float(len(ingram_fr_list))
        ingram_cr_rate = 100 * \
            sum([val for val in ingram_cr_list]) / float(len(ingram_cr_list))

        outgram_fa_list = [val for val,
                           gram_status in fa_list if gram_status == 'OOG']
        outgram_cr_list = [val for val,
                           gram_status in cr_list if gram_status == 'OOG']
        outgram_fr_list = [val for val,
                           gram_status in fr_list if gram_status == 'OOG']

        outgram_fa_rate = 100 * \
            sum([val for val in outgram_fa_list]) / float(len(outgram_fa_list))
        outgram_cr_rate = 100 * \
            sum([val for val in outgram_cr_list]) / float(len(outgram_cr_list))
        outgram_fr_rate = 100 * \
            sum([val for val in outgram_fr_list]) / float(len(outgram_fr_list))

        return {'overall': (ca_rate, fa_rate, cr_rate, fr_rate),
                'ingram': (ingram_ca_rate, ingram_fa_rate, ingram_cr_rate, ingram_fr_rate),
                'outgram': (outgram_fa_rate, outgram_cr_rate, outgram_fr_rate)}

    def print_overall_metrics(filename, threshold=100):
        '''Prints all grammar metrics'''
        info = get_overall_metrics(filename, threshold=100)
        (ca_rate, fa_rate, cr_rate, fr_rate) = info['overall']
        (ingram_ca_rate, ingram_fa_rate, ingram_cr_rate,
         ingram_fr_rate) = info['ingram']
        (outgram_fa_rate, outgram_cr_rate,
         outgram_fr_rate) = info['outgram']

        data = (ca_rate, fa_rate, cr_rate, fr_rate)
        # print ca_rate , fa_rate , cr_rate , fr_rate
        print tabulate([data],
                       headers=['Correct accept rate',
                                'False accept rate',
                                'Correct reject rate',
                                'False reject rate'],
                       tablefmt='simple', numalign="center") + '\n'

        # print ingram_ca_rate , ingram_fa_rate , ingram_fr_rate ,
        # ingram_cr_rate
        data = (ingram_ca_rate, ingram_fa_rate, ingram_cr_rate, ingram_fr_rate)
        print '\n' * 2
        print tabulate([data],
                       headers=['In grammar correct accept rate',
                                'In grammar false accept rate',
                                'In grammar correct reject rate',
                                'In grammar false reject rate'],
                       tablefmt='simple', numalign="center") + '\n'

        # print outgram_fa_rate , outgram_cr_rate , outgram_fr_rate
        data = (outgram_fa_rate, outgram_cr_rate, outgram_fr_rate)
        print '\n' * 2
        print tabulate([data],
                       headers=['Out of grammar false accept rate',
                                'Out of grammar correct reject rate',
                                'Out of grammar false reject rate'],
                       tablefmt='simple', numalign="center") + '\n'

    def get_user_metrics(filename, threshold=100, day=None):
        '''returns metrics like FR, FA, CA, CR per user
        Input can be a .csv file or .interaction file.
        Assumes Lumenvox interactionfile csv (no headers and renamed to .csv) 
        or the entire Lumenvox interaction file with (untouched with headers 
        intact and .interaction extension)'''

        users_ca_dict, users_cr_dict, users_fa_dict, users_fr_dict, user_conf_dict = {}, {}, {}, {}, {}

        reader = get_reader(filename)
        for row in reader:
            (transcript_si, transcript, decode_si, decode, conf, decode_time,
             callsrepath, acoustic_model,
             date, time, milliseconds, grammarlevel, firstname, lastname, oration_id,
             chain, store) = process(row)
             
             

            ca, fa, cr, fr = (compute_ca(transcript_si, decode_si, conf, threshold),
                              compute_fa(
                                  transcript_si, decode_si, conf, threshold),
                              compute_cr(
                                  transcript_si, decode_si, conf, threshold),
                              compute_fr(transcript_si, decode_si, conf, threshold))

            users_ca_dict.setdefault(firstname + '_' + lastname, [])
            users_ca_dict[firstname + '_' + lastname].append(ca)

            users_cr_dict.setdefault(firstname + '_' + lastname, [])
            users_cr_dict[firstname + '_' + lastname].append(cr)

            users_fa_dict.setdefault(firstname + '_' + lastname, [])
            users_fa_dict[firstname + '_' + lastname].append(fa)

            users_fr_dict.setdefault(firstname + '_' + lastname, [])
            users_fr_dict[firstname + '_' + lastname].append(fr)

            user_conf_dict.setdefault(firstname + '_' + lastname, [])
            user_conf_dict[firstname + '_' + lastname].append(conf)

        # return users_ca_dict, users_cr_dict, users_fa_dict, users_fr_dict
        user_ca_rate_list = [(user, 100 * float(sum(vals)) / len(vals), len(vals))
                             for user, vals in users_ca_dict.items()]
        user_cr_rate_list = [(user, 100 * float(sum(vals)) / len(vals), len(vals))
                             for user, vals in users_cr_dict.items()]
        user_fa_rate_list = [(user, 100 * float(sum(vals)) / len(vals), len(vals))
                             for user, vals in users_fa_dict.items()]
        user_fr_rate_list = [(user, 100 * float(sum(vals)) / len(vals), len(vals))
                             for user, vals in users_fr_dict.items()]
        user_mean_conf_list = [(user, float(sum(vals)) / len(vals), len(vals))
                               for user, vals in user_conf_dict.items()]
        return user_ca_rate_list, user_cr_rate_list, user_fa_rate_list, user_fr_rate_list, user_mean_conf_list

    def print_user_metrics(filename, sort_by_metric='ca'):
        ''' Calls the get_user_metrics and displays it in a sorted manner'''
        csvfilename = filename

        (user_ca_rate_list, user_cr_rate_list,
         user_fa_rate_list, user_fr_rate_list, user_mean_conf_list) = get_user_metrics(csvfilename)

        headers = ['USER', 'Correct accept rate', 'Correct reject rate', 'False accept rate',
                   'False reject rate', 'Total error rate', 'Mean conf', 'Number of instances']

        users = list(set([user for user, _, _ in user_ca_rate_list +
                          user_cr_rate_list + user_fa_rate_list + user_fr_rate_list]))
        user_metrics_rows = []
        for user in users:

            if hasNumbers(user):
                continue

            for User, rate, num_instances in user_ca_rate_list:
                if user == User:
                    ca = rate
                    num = num_instances
                    break
            for User, rate, num_instances in user_cr_rate_list:
                if user == User:
                    cr = rate
                    break
            for User, rate, num_instances in user_fa_rate_list:
                if user == User:
                    fa = rate
                    break
            for User, rate, num_instances in user_fr_rate_list:
                if user == User:
                    fr = rate
                    break

            for User, conf, num_instances in user_mean_conf_list:
                if user == User:
                    mean_conf = conf
                    break
            ter = fa + fr
            user_metrics_rows.append(
                (user, ca, cr, fa, fr, ter, mean_conf, num))

        code = {'name': 0, 'ca': 1, 'cr': 2, 'fa': 3,
                'fr': 4, 'ter': 5, 'mean_conf': 6, 'num_instances': 7}
        type = code[sort_by_metric]
        print tabulate(sorted(user_metrics_rows, key=lambda x: x[type]),
                       headers=headers, tablefmt="simple", numalign="center") + '\n'

    def get_power_users(filename, num_users=10):
        '''Gets a list of users by highest usage count/occurences in 
        interaction file.'''
        csvfilename = filename
        user_ca_rate_list, _, _, _, _ = get_user_metrics(
            csvfilename)
        power_users = [name for name, _, _ in sorted(
            user_ca_rate_list, key=lambda x: x[2], reverse=True)[:num_users]]
        return filter(lambda user: not hasNumbers(user), power_users)

    def get_users_with_top_ca_rates(filename, num_users=10):
        '''Gets a list of users with highest correct accept rates'''
        csvfilename = filename
        user_ca_rate_list, _, _, _, _ = get_user_metrics(
            csvfilename)
        top_ca_users = [name for name, _, _ in sorted(
            user_ca_rate_list, key=lambda x: x[1], reverse=True)[:num_users]]
        return filter(lambda user: not hasNumbers(user), top_ca_users)

    def get_users_with_lowest_fa_rates(filename, num_users=10):
        '''Gets a list of users with lowest false accept rates'''
        csvfilename = filename
        _, _, user_fa_rate_list, _, _ = get_user_metrics(
            csvfilename)
        low_fa_users = [name for name, _, _ in sorted(
            user_fa_rate_list, key=lambda x: x[1])[:num_users]]
        return filter(lambda user: not hasNumbers(user), low_fa_users)

    def get_successfull_power_users(filename, with_FA_considered=True, num_users=35):
        '''Gets a list of power users. Gets a list of top CA rates user. Gets 
        a list of lowest FA rates. Returns an intersection of these three lists
        '''
        power_users = get_power_users(filename, num_users=num_users)
        top_ca_rates_users = get_users_with_top_ca_rates(
            filename, num_users=num_users)
        low_fa_rate_users = get_users_with_lowest_fa_rates(
            filename, num_users=num_users)

        if with_FA_considered:
            return list(sorted(list(set(power_users).intersection
                                    (set(top_ca_rates_users)).intersection
                                    (set(low_fa_rate_users))
                                    )
                               )
                        )
        else:
            return list(sorted(list(set(power_users).intersection
                                    (set(top_ca_rates_users))
                                    )
                               )
                        )

    def get_best_and_worst_ter_users(filename, num_users=10):
        '''Tries to find an intersection of users who are power users and 
        lowest total error rate. Also tries to find an intersection of users 
        who are power users and highest total error rate'''
        csvfilename = filename
        _, _, user_fa_rate_list, user_fr_rate_list, _ = get_user_metrics(
            csvfilename)

        user_ters = []
        for (user, fa, _), (_, fr, _) in zip(sorted(user_fa_rate_list, key=lambda x: x[0]), sorted(user_fr_rate_list, key=lambda x: x[0])):
            ter = fa + fr
            if not hasNumbers(user):
                user_ters.append((user, ter))

        # pick top num_users(say 10) users with best ter
        best_ter = list(sorted(user_ters, key=lambda x: x[1]))[:num_users]
        # pick bottom num_users(say 10) users with best ter
        worst_ter = list(sorted(user_ters, key=lambda x: x[1]))[-num_users:]

        best_ter_users = [user for user, ter in best_ter]
        worst_ter_users = [user for user, ter in worst_ter]
        power_users = get_power_users(filename, num_users=num_users)

        power_best_ter_users = list(
            set(power_users).intersection(set(best_ter_users)))
        power_worst_ter_users = list(
            set(power_users).intersection(set(worst_ter_users)))

        '''This might seem confusing. All it is doing is using the power user list and populating
        another list if a name in the power list is in best ter list to achieve a sort using number of instances of user'''
        power_best_ter_users_sorted = []
        for user in power_users:
            if user in power_best_ter_users:
                power_best_ter_users_sorted.append(user)
        '''This might seem confusing. All it is doing is using the power user list and populating
        another list if a name in the power list is in worst ter list to achieve a sort using number of instances of user'''
        power_worst_ter_users_sorted = []
        for user in power_users:
            if user in power_worst_ter_users:
                power_worst_ter_users_sorted.append(user)

        # Limits number of users to 5

        if len(power_best_ter_users_sorted) >= 5:
            power_best_ter_users_sorted = power_best_ter_users_sorted[:5]
        if len(power_worst_ter_users_sorted) >= 5:
            power_worst_ter_users_sorted = power_worst_ter_users_sorted[:5]

        return {'power_best_ter_users': power_best_ter_users_sorted, 'power_worst_ter_users': power_worst_ter_users_sorted}

    def get_metrics_change_with_thresholds(filename):
        thresholds = range(0, 500, 20)
        threshold_overall_metrics = []
        threshold_ingram_metrics = []
        threshold_outgram_metrics = []
        for threshold in thresholds:
            info = get_overall_metrics(filename, threshold=threshold)
            (ca_rate, fa_rate, cr_rate, fr_rate) = info['overall']
            (ingram_ca_rate, ingram_fa_rate, ingram_cr_rate,
             ingram_fr_rate) = info['ingram']
            (outgram_fa_rate, outgram_cr_rate,
             outgram_fr_rate) = info['outgram']

            threshold_overall_metrics.append(
                (threshold, ca_rate, fa_rate, cr_rate, fr_rate, fa_rate + fr_rate))
            threshold_ingram_metrics.append(
                (threshold, ingram_ca_rate, ingram_fa_rate, ingram_cr_rate, ingram_fr_rate, ingram_fa_rate + ingram_fr_rate))
            threshold_outgram_metrics.append(
                (threshold, outgram_fa_rate, outgram_cr_rate, outgram_fr_rate, outgram_fa_rate + outgram_fr_rate))

        return (threshold_overall_metrics, threshold_ingram_metrics, threshold_outgram_metrics)

    def print_metrics_change_with_thresholds(threshold_overall_metrics, threshold_ingram_metrics, threshold_outgram_metrics):

        print tabulate(threshold_overall_metrics, headers=['Threshold',
                                                           'Correct accept rate',
                                                           'False accept rate',
                                                           'Correct reject rate',
                                                           'False reject rate',
                                                           'Total error rate'],
                       tablefmt='simple', numalign="center") + '\n'

        print tabulate(threshold_ingram_metrics, headers=['Threshold',
                                                          'In grammar correct accept rate',
                                                          'In grammar false accept rate',
                                                          'In grammar correct reject rate',
                                                          'In grammar false reject rate',
                                                          'Total error rate'],
                       tablefmt='simple', numalign="center") + '\n'

        print tabulate(threshold_outgram_metrics, headers=['Threshold',
                                                           'Out of grammar false accept rate',
                                                           'Out of grammar correct reject rate',
                                                           'Out of grammar false reject rate',
                                                           'Total error rate'],
                       tablefmt='simple', numalign="center") + '\n'

    def get_transcript_tag_statistics(filename):

        tag_counter = 0
        clip_click_tag_counter = 0
        noise_tag_counter = 0
        background_speech_tag_counter = 0
        total = 0
        for row in get_reader(filename):
            total = total + 1
            (transcript_si, transcript, decode_si, decode, conf,
                decode_time, callsrepath, acoustic_model, date, time,
                milliseconds, grammarlevel, firstname, lastname,
                oration_id, chain, store) = process(row)
            if '++' in transcript:
                tag_counter = tag_counter + 1
            if 'CLIP' in transcript or 'CLICK' in transcript:
                clip_click_tag_counter = clip_click_tag_counter + 1
            if 'NOISE' in transcript:
                noise_tag_counter = noise_tag_counter + 1
            if 'SPEECH' in transcript:
                background_speech_tag_counter = background_speech_tag_counter + 1

        return (tag_counter, clip_click_tag_counter, noise_tag_counter, background_speech_tag_counter, total)

    def print_transcript_tag_statistics(tag_counter, clip_click_tag_counter, noise_tag_counter, background_speech_tag_counter, total):
        # print tabulate([[100 * float(tag_counter) / total, 100 *
        # float(clip_click_tag_counter) / total, 100 * float(noise_tag_counter)
        # / total, 100 * float(background_speech_tag_counter) / total]],
        # headers=['All tags percentage', 'Clip Click percentage', 'Noise
        # percentage', 'Background Speech percentage'], tablefmt='simple',
        # numalign='center')

        print
        print '{:<30}'.format('All tags percentage') + ": " + str(100 * float(tag_counter) / total)
        print '{:<30}'.format('Clip Click percentage') + ": " + str(100 * float(clip_click_tag_counter) / total)
        print '{:<30}'.format('Noise percentage') + ": " + str(100 * float(noise_tag_counter) / total)
        print '{:<30}'.format('Background Speech percentage') + ": " + str(100 * float(background_speech_tag_counter) / total)

    def get_grammar_statistics(filename):

        grammar_statistics = {
            'G1': {'count': 0, 'confs': []}, 'G2': {'count': 0, 'confs': []}}
        for row in get_reader(filename):
            (transcript_si, transcript, decode_si, decode, conf,
                decode_time, callsrepath, acoustic_model, date, time,
                milliseconds, grammarlevel, firstname, lastname,
                oration_id, chain, store) = process(row)

            if grammarlevel == 'G1':
                grammar_statistics['G1']['count'] += 1
                grammar_statistics['G1']['confs'].append(int(conf))
            else:
                grammar_statistics['G2']['count'] += 1
                grammar_statistics['G2']['confs'].append(int(conf))

        return grammar_statistics

    def print_grammar_statistics(grammar_statistics):
        print
        print '{:<30}'.format('G1 count') + ": " + str(grammar_statistics['G1']['count'])
        print '{:<30}'.format('G2 count') + ": " + str(grammar_statistics['G2']['count'])
        print '{:<30}'.format('G1 mean confidence') + ": " + str(sum(grammar_statistics['G1']['confs']) / float(len(grammar_statistics['G1']['confs'])))
        print '{:<30}'.format('G2 mean confidence') + ": " + str(sum(grammar_statistics['G2']['confs']) / float(len(grammar_statistics['G2']['confs'])))

    def get_command_confs(filename):
        commands_conf = {}
        for row in get_reader(filename):
            (transcript_si, transcript, decode_si, decode, conf,
                decode_time, callsrepath, acoustic_model, date, time,
                milliseconds, grammarlevel, firstname, lastname,
                oration_id, chain, store) = process(row)

            command = transcript_si.split(';')[0]
            commands_conf.setdefault(command, [])
            commands_conf[command].append(int(conf))

        return commands_conf

    def print_commands_conf(command_confs):
        command_mean_conf = [(command, sum(confs) / float(len(confs)), len(confs))
                             for command, confs in command_confs.items()]

        command_mean_conf = sorted(command_mean_conf, key=lambda x: x[2])

        print
        print tabulate(command_mean_conf, headers=['Commands', 'Mean Confidence', 'Number of Instances'],
                       tablefmt='simple', numalign="center") + '\n'

    def clean_up_phrase(phrase):
        p = re.compile(ur'(\+\+.*\+\+)')
        result = re.search(p, phrase)
        temp = phrase
        if result:
            for item in result.groups()[0].split(' '):
                temp = temp.replace(item, '')

        return temp

    def get_OOV_words(filename):

        oov_words = []
        for row in get_reader(filename):
            (transcript_si, transcript, decode_si, decode, conf,
                decode_time, callsrepath, acoustic_model, date, time,
                milliseconds, grammarlevel, firstname, lastname,
                oration_id, chain, store) = process(row)

            if transcript_si in ['~No interpretations']:
                oov_words.append(transcript)

        oog_phrase_counter = dict(
            Counter([clean_up_phrase(phrase) for phrase in oov_words]))
        oog_counter = dict(
            Counter(filter(lambda x: '++' not in x, (' '.join(oov_words).split(' ')))))
        return oog_counter, oog_phrase_counter

    def print_oog_word_count(oog_counter, oog_phrase_counter):
        sorted_oog_counts = [(oog_word, count) for oog_word, count in sorted(
            oog_counter.items(), key=lambda x: x[1], reverse=True)]

        # This is because the same oog doesn't said over and over agian
        print
        print tabulate(sorted_oog_counts, headers=['OOG word', 'Count'],
                       tablefmt='simple', numalign="center") + '\n'
        print oog_phrase_counter  # Nothing to aggregate and show in this.

    def get_ingram_outgram_percentage(filename):

        oov_count = 0
        total = 0
        for row in get_reader(filename):
            (transcript_si, transcript, decode_si, decode, conf,
                decode_time, callsrepath, acoustic_model, date, time,
                milliseconds, grammarlevel, firstname, lastname,
                oration_id, chain, store) = process(row)
            total += 1
            if transcript_si in ['~No interpretations']:
                oov_count += 1

        oog_percent = 100 * float(oov_count) / total
        ing_percent = 100 - oog_percent

        return {'ing_percent': ing_percent, 'oog_percent': oog_percent}

    def print_ingram_out_gram(ing_percent=100, oog_percent=0):
        print
        print '{:<30}'.format('In grammar percentage') + ": " + str(ing_percent)
        print '{:<30}'.format('Out of grammar percentage') + ": " + str(oog_percent)
        print
        
    def get_dates(filename):
        expr = ur'(2016|2017|2018|2019)([0-1][0-9])([0-3][0-9])'
        p = re.compile(expr)
        dates_list = []
        for row in get_reader(filename):
            (transcript_si, transcript, decode_si, decode, conf,
         decode_time, callsrepath, acoustic_model, date, time,
         milliseconds, grammarlevel, firstname, lastname,
         oration_id, chain, store) = process(row)
               
            result = re.search(p, date)
            year, month, day = map(int, result.groups()[:3])
            dates_list.append((year, month, day))

        sorted_dates =  sorted(list(set(dates_list)), key=lambda x: 365*x[0] + 30*x[1] + x[2])
        return sorted_dates

    def get_metrics_per_day(filename):
        sorted_dates = get_dates(filename)
        date_overall_metrics = []
        date_ingram_metrics = []
        for year, month, day in sorted_dates:
            info = get_overall_metrics(filename, threshold=100, day=day)
            (ca_rate, fa_rate, cr_rate, fr_rate) = info['overall']
            (ingram_ca_rate, ingram_fa_rate, ingram_cr_rate,
            ingram_fr_rate) = info['ingram']
            
            
            weekday = calendar.day_name[date(year, month, day).weekday()]
            date_str = '%s-%s-%s' % (month, day, year)
            date_overall_metrics.append((weekday, date_str, ca_rate, fa_rate, cr_rate, fr_rate, fa_rate + fr_rate))
            
            date_ingram_metrics.append((weekday, date_str, ingram_ca_rate, ingram_fa_rate, ingram_cr_rate,
            ingram_fr_rate, ingram_fr_rate + ingram_fa_rate))
         
        return date_overall_metrics, date_ingram_metrics 

    def print_metrics_per_day(date_overall_metrics, date_ingram_metrics):
     
        print tabulate(date_overall_metrics,
                           headers=['Weekday', 'Date',
                                    'Correct accept rate',
                                    'False accept rate',
                                    'Correct reject rate',
                                    'False reject rate',
                                    'Total error rate'],
                           tablefmt='simple', numalign="center") + '\n'
                           
        print            
        print tabulate(date_ingram_metrics,
                           headers=['Weekday', 'Date',
                                    'In gram Correct accept rate',
                                    'In gram False accept rate',
                                    'In gram Correct reject rate',
                                    'In gram False reject rate',
                                    'Total error rate'],
                           tablefmt='simple', numalign="center") + '\n'
    
    
    def get_times(filename):
        for row in get_reader(filename):
            (transcript_si, transcript, decode_si, decode, conf,
         decode_time, callsrepath, acoustic_model, date, time,
         milliseconds, grammarlevel, firstname, lastname,
         oration_id, chain, store) = process(row)
        
    
    def main():
        print 'The data source has been set to csv\n'
        #filename = 'MIC-LEW_20160220-0229_all.csv'
        filename = 'data/test1.interactions'
        filename = 'data/converted.csv'
        filename = 'data/MIC-LEW_20160220-0229_all.Interactions'
        #filename = 'data/TCS-AUS_20150905_ALL.Interactions'
        # get_user_metrics(filename)
        
        '''
        print 'Successful power users according one criteria are', ', '.join(get_successfull_power_users(filename, with_FA_considered=True, num_users=40))
        power_best_worst_ter_users = get_best_and_worst_ter_users(
            filename, num_users=20)
        print 'Sucessfull power users according to another criteria (TER) are', ', '.join(power_best_worst_ter_users['power_best_ter_users'])
        print 'Struggling power users according to (TER) are', ', '.join(power_best_worst_ter_users['power_worst_ter_users'])
        print

        print_transcript_tag_statistics(
            *get_transcript_tag_statistics(filename))

        print_grammar_statistics(get_grammar_statistics(filename))
        print_commands_conf(get_command_confs(filename))
        print_ingram_out_gram(**get_ingram_outgram_percentage(filename))

        print_user_metrics(filename, sort_by_metric='ter')

        print_metrics_per_day(*get_metrics_per_day(filename))
        print_metrics_change_with_thresholds(
            *get_metrics_change_with_thresholds(filename))

        print_oog_word_count(*get_OOV_words(filename))
        '''
        get_times(filename)
        

if __name__ == '__main__':
    main()
