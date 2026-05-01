def fill_with_spaces(input_str, length):
    return f"{input_str:<{length}}" 

multi = []
with open('multi_layer.txt') as file:
    multi = file.readlines()

single = []
with open('single_layer.txt') as file:
    single = file.readlines()

multi_def = []
with open('multi_layer_default.txt') as file:
    multi_def = file.readlines()

for i in range( len(single)):
    zoz = []
    zoz.append(fill_with_spaces(single[i][single[i].find('=') +1 :].strip(), 6))
    zoz.append(fill_with_spaces(multi_def[i][multi_def[i].find('=') + 1:].strip(), 6))
    zoz.append(fill_with_spaces(multi[i][multi[i].find('=') + 1:].strip(), 6))
    
    print(f"Generation: {fill_with_spaces(str(i+1), 3)}  ||", end=" ")
    print(*zoz, sep='|  ')