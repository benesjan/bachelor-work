import config

data_path = config.data['test']

num_articles = 0
num_lines = 0
with open(data_path, 'r', encoding='utf-8') as handler:
    for line in handler:
        if line.startswith('<article'):
            num_articles += 1
        elif line != '</article>\n':
            num_lines += 1

print("Average number of paragraphs within article: %.2f" % (num_lines/num_articles))