def get_rows(interactionsfilename):
    with open(interactionsfilename, 'r') as f:
        rows = [line for line in f]
        rows = filter(lambda row: len(row) > 100, rows)
        return rows


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
     ) = row.split(',')

    date, time, milliseconds, grammarlevel, fullname, oration_id, chain, store = callsrepath.split(
        '\\')[-1].split('_')

    try:
        firstname, lastname = fullname.split(' ')
    except:
        firstname, lastname = fullname, fullname

    return (transcript_si, transcript, decode_si, decode, conf, decode_time,
            callsrepath, acoustic_model,
            date, time, milliseconds, grammarlevel, firstname, lastname, oration_id,
            chain, store)
