# read lines from all
lines = []
with open("udemy_all.txt", "r") as file:
    lines = file.read().splitlines()

# split and get sentences
sentences = []
for line in lines:
    for sentence in line.split('。'):
        if len(sentence)>5:
            sentences.append(sentence.replace('■', '').replace('・', '').replace('■', '').replace('▲', '').replace('△', '').replace('▼', '').replace('▽', '').replace('◆', '').replace('◇', '').replace('○', '').replace('◎', '').replace('●', '').replace('★', '').replace('☆', '').replace(' ', ''))

# get feelings list
feelings_bad = []
with open("feelings_bad.txt", "r") as file:
    feelings_bad = file.read().splitlines()

feelings_good = []
with open("feelings_good.txt", "r") as file:
    feelings_good = file.read().splitlines()

# select feelings from all
good_selected = []
bad_selected = []
for sentence in sentences:
    good = False
    for feeling_good in feelings_good:
        if feeling_good in sentence:
            good = True
            good_selected.append(sentence)
            break
    if not good:
        for feeling_bad in feelings_bad:
            if feeling_bad in sentence:
                bad_selected.append(sentence)
                break

# remove duplicated
filtered_good = list(set(good_selected))
filtered_bad = list(set(bad_selected))

# save selected
with open('udemy_good.txt', 'w+') as file:
    for review in filtered_good:
        file.write('%s\n' % review)

print(len(filtered_good))

with open('udemy_bad.txt', 'w+') as file:
    for review in filtered_bad:
        file.write('%s\n' % review)

print(len(filtered_bad))

# print('writing removed file...')
# # save removed
# with open('udemy_removed.txt', 'w+') as file:
#     for review in sentences:
#         if review in filtered_selected:
#             continue
#         print('.', end='')
#         file.write('%s\n' % review)
