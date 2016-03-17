import dbhelper
import orm_helper
import os
import math
from constants import DATA_DIR, DB_PATH, DATA_SOURCE

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

        return (transcript_si, transcript, decode_si, decode, conf,
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

    def get_overall_metrics(filename, threshold=100):
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

    def get_user_metrics(filename, threshold=100):
        '''returns metrics like FR, FA, CA, CR per user
        Input can be a .csv file or .interaction file.
        Assumes Lumenvox interactionfile csv (no headers and renamed to .csv) 
        or the entire Lumenvox interaction file with (untouched with headers 
        intact and .interaction extension)'''

        users_ca_dict, users_cr_dict, users_fa_dict, users_fr_dict = {}, {}, {}, {}

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

        # return users_ca_dict, users_cr_dict, users_fa_dict, users_fr_dict
        user_ca_rate_list = [(user, 100 * float(sum(vals)) / len(vals), len(vals))
                             for user, vals in users_ca_dict.items()]
        user_cr_rate_list = [(user, 100 * float(sum(vals)) / len(vals), len(vals))
                             for user, vals in users_cr_dict.items()]
        user_fa_rate_list = [(user, 100 * float(sum(vals)) / len(vals), len(vals))
                             for user, vals in users_fa_dict.items()]
        user_fr_rate_list = [(user, 100 * float(sum(vals)) / len(vals), len(vals))
                             for user, vals in users_fr_dict.items()]

        return user_ca_rate_list, user_cr_rate_list, user_fa_rate_list, user_fr_rate_list

    def print_user_metrics(filename, sort_by_metric='ca'):
        ''' Calls the get_user_metrics and displays it in a sorted manner'''
        csvfilename = filename
        
        (user_ca_rate_list, user_cr_rate_list, 
        user_fa_rate_list, user_fr_rate_list) = get_user_metrics(csvfilename)

        headers = ['USER', 'Correct accept rate', 'Correct reject rate', 'False accept rate',
                   'False reject rate', 'Total error rate', 'Number of instances']

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
            ter = fa + fr
            user_metrics_rows.append((user, ca, cr, fa, fr, ter, num))

        code = {'name': 0, 'ca': 1, 'cr': 2, 'fa': 3,
                'fr': 4, 'ter': 5, 'num_instances': 6}
        type = code[sort_by_metric]
        print tabulate(sorted(user_metrics_rows, key=lambda x: x[type]),
        headers=headers, tablefmt="simple", numalign="center") + '\n'

    def get_power_users(filename, num_users=10):
        '''Gets a list of users by highest usage count/occurences in 
        interaction file.'''
        csvfilename = filename
        user_ca_rate_list, _, _, _ = get_user_metrics(
            csvfilename)
        power_users = [name for name, _, _ in sorted(
            user_ca_rate_list, key=lambda x: x[2], reverse=True)[:num_users]]
        return filter(lambda user: not hasNumbers(user), power_users)

    def get_users_with_top_ca_rates(filename, num_users=10):
        '''Gets a list of users with highest correct accept rates'''
        csvfilename = filename
        user_ca_rate_list, _, _, _ = get_user_metrics(
            csvfilename)
        top_ca_users = [name for name, _, _ in sorted(
            user_ca_rate_list, key=lambda x: x[1], reverse=True)[:num_users]]
        return filter(lambda user: not hasNumbers(user), top_ca_users)

    def get_users_with_lowest_fa_rates(filename, num_users=10):
        '''Gets a list of users with lowest false accept rates'''
        csvfilename = filename
        _, _, user_fa_rate_list, _ = get_user_metrics(
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
        _, _, user_fa_rate_list, user_fr_rate_list = get_user_metrics(
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
          
    def how_metrics_change_with_thresholds(filename):
        '''TODO Move printing to a separate method'''
        thresholds = range(0, 500, 25)
        threshold_overall_metrics = []
        threshold_ingram_metrics = []
        threshold_outgram_metrics = []
        for threshold in thresholds:
            info = get_overall_metrics(filename, threshold=threshold)
            (ca_rate, fa_rate, cr_rate, fr_rate) = info['overall']
            (ingram_ca_rate, ingram_fa_rate, ingram_cr_rate, ingram_fr_rate) = info['ingram']
            (outgram_fa_rate, outgram_cr_rate, outgram_fr_rate) = info['outgram']
        
            threshold_overall_metrics.append((threshold, ca_rate, fa_rate, cr_rate, fr_rate, fa_rate + fr_rate))
            threshold_ingram_metrics.append((threshold, ingram_ca_rate, ingram_fa_rate, ingram_cr_rate, ingram_fr_rate, ingram_fa_rate + ingram_fr_rate))
            threshold_outgram_metrics.append((threshold, outgram_fa_rate, outgram_cr_rate, outgram_fr_rate, outgram_fa_rate + outgram_fr_rate))
            
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

    def transcript_tag_statistics(filename):
        
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
                
        print tabulate([[100*float(tag_counter)/total, 100*float(clip_click_tag_counter)/total, 100*float(noise_tag_counter)/total, 100*float(background_speech_tag_counter)/total]], headers=['All tags percentage', 'Clip Click percentage', 'Noise percentage', 'Background Speech percentage'], tablefmt='simple', numalign='center')

    def main():
        print 'The data source has been set to csv\n'
        #filename = 'MIC-LEW_20160220-0229_all.csv'
        filename = 'data/test1.interactions'
        filename = 'data/converted.csv'
        filename = 'data/MIC-LEW_20160220-0229_all.Interactions'
        #filename = 'data/TCS-AUS_20150905_ALL.Interactions'
        # get_user_metrics(filename)
        
        print 'Successful power users according one criteria are', ', '.join(get_successfull_power_users(filename, with_FA_considered=True, num_users=40))
        power_best_worst_ter_users = get_best_and_worst_ter_users(filename, num_users=20)
        print 'Sucessfull power users according to another criteria (TER) are', ', '.join(power_best_worst_ter_users['power_best_ter_users'])
        print 'Struggling power users according to (TER) are', ', '.join(power_best_worst_ter_users['power_worst_ter_users'])
        print 
        
        print_user_metrics(filename, sort_by_metric='ter')
        #get_overall_metrics(filename, threshold=100)
        #print_overall_metrics(filename, threshold=100)
        how_metrics_change_with_thresholds(filename)
        transcript_tag_statistics(filename)


if __name__ == '__main__':
    main()
