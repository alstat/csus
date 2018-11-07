import pandas as pd; import numpy as np
def mba(fname, nrows=1000,cols=['POST_DATE','PANALOKARD_NO','TRAN_DESC','AGE','GENDER'],
        grp_cols=['POST_DATE','PANALOKARD_NO','AGE','GENDER'], age_col='AGE', age_grp_col='AGE_GROUP', transaction_col='TRAN_DESC', maxv=100, grouping=[20,40,60],threshold_support=0.05,threshold_confidence=0.05,threshold_lift=0.05):
    if nrows=='default' or nrows<=0: df=pd.read_csv(fname,low_memory=False)
    else: df=pd.read_csv(fname,low_memory=False,nrows=nrows)

    df_new=df[cols]
    df_new=df_new.groupby(by=grp_cols).agg(lambda x: tuple(x)).applymap(np.unique).reset_index() # https://stackoverflow.com/questions/46275765/pandas-merge-row-data-with-multiple-values-to-python-list-for-a-column
    
    # convert age to age group
    lim=[0]+grouping+[maxv]; group_labels=[]
    for i in range(len(lim)-1):
        if i==0: group_labels.append('<'+str(lim[i+1]))
        elif i==len(lim)-2: group_labels.append(str(lim[i])+"+")
        else: group_labels.append(str(lim[i])+'-'+str(lim[i+1]))

    age_grp=pd.cut(df_new['AGE'],lim,labels=group_labels) # https://www.coursera.org/lecture/data-visualization/python-lesson-4-grouping-values-within-individual-variables-aXOVG                
    df_new[age_col]=age_grp # https://stackoverflow.com/questions/30226503/sort-ages-into-certain-age-groups-using-function
    df_new=df_new.rename(columns={age_col: age_grp_col}) # https://stackoverflow.com/questions/20868394/changing-a-specific-column-name-in-pandas-dataframe
    

    import itertools
    # https://github.com/amitkaps/machine-learning/blob/master/cf_mba/notebook/2.%20Market%20Basket%20Analysis.ipynb
    def frequent_itemsets(sentences,threshold=1):
        # Counts sets with Apriori algorithm.
        SUPP_THRESHOLD = threshold
        supps = []

        supp = {}
        for sentence in sentences:
            for key in sentence:
                if key in supp: supp[key] += 1
                else: supp[key] = 1
        supps.append({k:v for k,v in supp.items() if v >= SUPP_THRESHOLD})

        supp = {}
        for sentence in sentences:
            for combination in itertools.combinations(sentence, 2):
                if combination[0] in supps[0] and combination[1] in supps[0]:
                    key = ','.join(combination)
                    if key in supp: supp[key] += 1
                    else: supp[key] = 1

        supps.append({k:v for k,v in supp.items() if v >= SUPP_THRESHOLD})

        supp = {}
        for sentence in sentences:
            for combination in itertools.combinations(sentence, 3):
                if (combination[0]+','+combination[1] in supps[1] and
                        combination[0]+','+combination[2] in supps[1] and
                        combination[1]+','+combination[2] in supps[1]):
                    key = ','.join(combination)
                    if key in supp: supp[key] += 1
                    else: supp[key] = 1

        supps.append({k:v for k,v in supp.items() if v >= SUPP_THRESHOLD})

        return supps


    def measures(supp_ab, supp_a, supp_b, transaction_count):
        # Assumes A -> B, where A and B are sets.
        conf = float(supp_ab) / float(supp_a)
        s = float(supp_b) / float(transaction_count)
        lift = conf / s
        return [conf, lift]

    def generate_rules(measure, supps, transaction_count,min_confidence=threshold_confidence,min_lift=threshold_lift):
        rules = []
        CONF_THRESHOLD = min_confidence
        LIFT_THRESHOLD = min_lift
        if measure == 'conf':
            for i in range(2, len(supps)+1):
                for k,v in supps[i-1].items():
                    k = k.split(',')
                    for j in range(1, len(k)):
                        for a in itertools.combinations(k, j):
                            b = tuple([w for w in k if w not in a])
                            [conf, lift] = measures(v,
                                    supps[len(a)-1][','.join(a)],
                                    supps[len(b)-1][','.join(b)],
                                    transaction_count)
                            if conf >= CONF_THRESHOLD:
                                rules.append((a, b, conf, lift))
                rules = sorted(rules, key=lambda x: (x[0], x[1]))
                rules = sorted(rules, key=lambda x: (x[2]), reverse=True)
        elif measure == 'lift':
            for i in range(2, len(supps)+1):
                for k,v in supps[i-1].items():
                    k = k.split(',')
                    for j in range(1, len(k)):
                        for a in itertools.combinations(k, j):
                            b = tuple([w for w in k if w not in a])
                            [conf, lift] = measures(v,
                                    supps[len(a)-1][','.join(a)],
                                    supps[len(b)-1][','.join(b)],
                                    transaction_count)
                            if lift >= LIFT_THRESHOLD:
                                rules.append((a, b, conf, lift))
                rules = sorted(rules, key=lambda x: (x[0], x[1]))
                rules = sorted(rules, key=lambda x: (x[3]), reverse=True)
        else:
            for i in range(2, len(supps)+1):
                for k,v in supps[i-1].items():
                    k = k.split(',')
                    for j in range(1, len(k)):
                        for a in itertools.combinations(k, j):
                            b = tuple([w for w in k if w not in a])
                            [conf, lift] = measures(v,
                                    supps[len(a)-1][','.join(a)],
                                    supps[len(b)-1][','.join(b)],
                                    transaction_count)
                            if (conf >= CONF_THRESHOLD and
                                    lift >= LIFT_THRESHOLD):
                                rules.append((a, b, conf, lift))
                rules = sorted(rules, key=lambda x: (x[0], x[1]))
                rules = sorted(rules, key=lambda x: (x[2],x[3]), reverse=True)
        return rules


    sentences=df_new[transaction_col].values.tolist(); thresh=int(len(sentences)*threshold_support)
    supps=frequent_itemsets(sentences,threshold=thresh)
    rules = generate_rules('all', supps, len(sentences),min_confidence=threshold_confidence,min_lift=threshold_lift)
    for rule in rules:
        print (("{{{}}} -> {{{}}}, "
               "conf = {:.2f}, lift = {:.2f}").format(
              ', '.join(rule[0]), ', '.join(rule[1]), rule[2], rule[3]))        