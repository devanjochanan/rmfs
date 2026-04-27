to_print = ""
for i in range(53):
    one_row = []
    for j in range(53):
        if j > 9 and j < 49:
            if i % 3 == 0:
                one_row.append('0')
            else:
                if j > 44:
                    one_row.append('0')
                else:
                    if j % 5 == 0:
                        one_row.append('0')
                    else:
                        one_row.append('1')
        else:
            one_row.append('0')
    to_print += "{}\n".format(",".join(one_row))

with open('pod.csv', "w+") as outlet_json:
    print(to_print, file=outlet_json)
    
