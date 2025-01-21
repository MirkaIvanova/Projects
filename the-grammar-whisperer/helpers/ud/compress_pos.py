import ast
# ['root', 'amod', 'obj', 'case', 'amod', 'obl',.
 
# ADJ	    A
# ADP	    1
# ADV	    2
# AUX     3
# CCONJ   C
# DET     D
# SYM     S
# INTJ	I	
# NOUN    N
# NUM	    4
# PART    P
# PROPN   5
# PUNCT	6
# VERB    V		
# PRON	7
# SCONJ	
# X       X
 def compress_pos(pos_string):
    # Convert string representation of list to actual list
    pos_list = ast.literal_eval(pos_string) 
    
    pos_map = {
        'root': '1',
        'amod': '2',
        'obj':  '3',
        'case': '4'
    }
    
    # Convert using the mapping
    return ''.join(gender_map[g] for g in gender_list)

# Example usage:
sequence = ['', '', 'M', '', '', '', 'F', 'F', '', 'N', '', '', '', 'F', '', 'N', '', 'N', '', 'F', 'N', 'N', '']
print(compress_gender(sequence)) 