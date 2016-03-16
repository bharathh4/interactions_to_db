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

# say the interaction file itself
elif DATA_SOURCE is 'csv':

    import csv

    def interaction2csv(interactionfilename, output_csvfilename):
        with open(interactionfilename, 'r') as fread:
            str_to_write = ''.join([line for line in fread if len(line) > 100])
            with open(output_csvfilename, 'w') as foutput:
                foutput.write(str_to_write)

    def get_reader(filename):
        # Gets the reader object for a csv/interaction file

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

        date, time, milliseconds, grammarlevel, fullname, oration_id, chain, store = callsrepath.split(
            '\\')[-1].split('_')

        try:
            firstname, lastname = fullname.split(' ')
        except:
            firstname, lastname = fullname, fullname

        return (transcript_si, transcript, decode_si, decode, conf, decode_time,
                callsrepath, acoustic_model,
                date, time, milliseconds, grammarlevel, firstname, lastname, oration_id,
                chain, store.replace('.callsre', ''))

    def compute_cain(transcript, decode, conf, threshold):

        if transcript == decode and int(conf) >= threshold:
            return True
        else:
            return False

    def compute_frin(transcript, decode, conf, threshold):
        if transcript == decode and int(conf) < threshold:
            return True
        else:
            return False

    def compute_fain(transcript, decode, conf, threshold):
        if transcript != decode and int(conf) >= threshold:
            return True
        else:
            return False

    def compute_crin(transcript, decode, conf, threshold):
        if transcript != decode and int(conf) < threshold:
            return True
        else:
            return False

    def get_user_metrics(filename):
        # returns metrics like FR, FA, CA, CR for all users
        # Input can be a .csv file or .interaction file.
        # Assumes Lumenvox interactionfile csv (no headers and renamed to .csv) or the entire Lumenvox interaction file with
        # (untouched with headers intact and .interaction extension)

        users_ca_dict, users_cr_dict, users_fa_dict, users_fr_dict = {}, {}, {}, {}

        reader = get_reader(filename)
        for row in reader:
            (transcript_si, transcript, decode_si, decode, conf, decode_time,
             callsrepath, acoustic_model,
             date, time, milliseconds, grammarlevel, firstname, lastname, oration_id,
             chain, store) = process(row)
            
            threshold = 100
            ca, fa, cr, fr = (compute_cain(transcript_si, decode_si, conf, threshold),
                              compute_fain(
                                  transcript_si, decode_si, conf, threshold),
                              compute_crin(
                                  transcript_si, decode_si, conf, threshold),
                              compute_frin(transcript_si, decode_si, conf, threshold))

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
        # Calls the get_user_metrics and displays it
        csvfilename = filename
        user_ca_rate_list, user_cr_rate_list, user_fa_rate_list, user_fr_rate_list = get_user_metrics(
            csvfilename)
            
        headers = ['USER', 'Correct accept rate', 'Correct reject rate', 'False accept rate', 'False reject rate', 'Total error rate', 'Number of instances']
        
        users = list(set([user for user, _, _ in user_ca_rate_list + user_cr_rate_list + user_fa_rate_list + user_fr_rate_list]))
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
            
        code = {'name':0, 'ca':1, 'cr':2, 'fa':3, 'fr':4, 'ter':5, 'num_instances':5}
        type = code[sort_by_metric]   
        print tabulate(sorted(user_metrics_rows, key=lambda x: x[type]), headers=headers)    
                
    def get_power_users(filename, num_users=10):
        csvfilename = filename
        user_ca_rate_list, _, _, _ = get_user_metrics(
            csvfilename)
        power_users = [name for name, _, _ in sorted(user_ca_rate_list, key=lambda x: x[2], reverse=True)[:num_users]]
        return filter(lambda user: not hasNumbers(user), power_users)
        
    def get_users_with_top_ca_rates(filename, num_users=10):
        csvfilename = filename
        user_ca_rate_list, _, _, _ = get_user_metrics(
            csvfilename)
        top_ca_users = [name for name, _, _ in sorted(user_ca_rate_list, key=lambda x: x[1], reverse=True)[:num_users]]
        return filter(lambda user: not hasNumbers(user), top_ca_users)
                
    def get_users_with_lowest_fa_rates(filename, num_users=10):
        csvfilename = filename
        _, _, user_fa_rate_list, _ = get_user_metrics(
            csvfilename)
        low_fa_users = [name for name, _, _ in sorted(user_fa_rate_list, key=lambda x: x[1])[:num_users]]
        return filter(lambda user: not hasNumbers(user), low_fa_users)

    def get_successfull_power_users(filename, with_FA_considered=True, num_users=35):
    
    
        power_users = get_power_users(filename, num_users=num_users)
        top_ca_rates_users = get_users_with_top_ca_rates(filename, num_users=num_users)
        low_fa_rate_users = get_users_with_lowest_fa_rates(filename, num_users=num_users)
        
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
        csvfilename = filename
        _, _, user_fa_rate_list, user_fr_rate_list = get_user_metrics(
            csvfilename)
        
        
        user_ters = []
        for (user, fa, _), (_, fr, _) in zip(sorted(user_fa_rate_list, key=lambda x: x[0]), sorted(user_fr_rate_list, key=lambda x: x[0])):
            ter = fa + fr
            if not hasNumbers(user):
                user_ters.append((user, ter))
                                      
        best_ter = list(sorted(user_ters, key=lambda x: x[1]))[:num_users] # pick top num_users(say 10) users with best ter
        worst_ter = list(sorted(user_ters, key=lambda x: x[1]))[-num_users:] # pick bottom num_users(say 10) users with best ter
        
        best_ter_users = [user for user, ter in best_ter]
        worst_ter_users = [user for user, ter in worst_ter]
        power_users = get_power_users(filename, num_users=num_users) 
        
        power_best_ter_users = list(set(power_users).intersection(set(best_ter_users)))
        power_worst_ter_users = list(set(power_users).intersection(set(worst_ter_users)))
        
        return {'power_best_ter_users':power_best_ter_users, 'power_worst_ter_users': power_worst_ter_users}
            
    
    def hasNumbers(inputString):
        return any(char.isdigit() for char in inputString)
        
    def main():
        print 'The data source has been set to csv'
        #filename = 'MIC-LEW_20160220-0229_all.csv'
        filename = 'data/test1.interactions'
        filename = 'data/converted.csv'
        filename = 'data/MIC-LEW_20160220-0229_all.Interactions'
        # get_user_metrics(filename)
        print_user_metrics(filename, sort_by_metric='ter')
        print 'Successful power users are ', get_successfull_power_users(filename, with_FA_considered=False, num_users=35)
        print get_best_and_worst_ter_users(filename, num_users=15)


if __name__ == '__main__':
    main()
