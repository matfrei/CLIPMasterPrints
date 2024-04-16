from nltk.corpus import wordnet
superclasses = ["dog", "musical_instrument", "motor_vehicle", "vegetable"]

syn = wordnet.synsets("dog")[0]

for superclass in superclasses:
    syn = wordnet.synsets(superclass)[0]
    print(syn.name())
    hyponyms = syn.hyponyms()
    hyponyms = [lemma.name() for hn in syn.hyponyms() for lemma in hn.lemmas()]
    hyponyms = [class_str.replace('_',' ') for class_str in hyponyms]
    print(hyponyms)

    # as more closely related hyponyms occur in sequence, the split that ensures the most homogenity  \
    # (without actually comparing similarity scores) just splits the already ordered elements in the middle

    split_idx = int(len(hyponyms)/2)
    hyponyms_opt = hyponyms[:split_idx]
    hyponyms_nonopt = hyponyms[split_idx:]
    opt_str = 'opt'
    filepath = f'./{superclass}_{opt_str}.txt'
    
    print(hyponyms_opt)
    print(hyponyms_nonopt)

    with open(filepath,'w') as outfile:
        for element in hyponyms_opt:
            outfile.write(f'{element}\n')

    opt_str = 'nonopt'
    filepath = f'./{superclass}_{opt_str}.txt'
            
    with open(filepath,'w') as outfile:
        for element in hyponyms_nonopt:
            outfile.write(f'{element}\n')
