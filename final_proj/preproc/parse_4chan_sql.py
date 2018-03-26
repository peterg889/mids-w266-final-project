
from csv import reader
from StringIO import StringIO 
import re
import os

# file = open("/Users/petergrabowski/Downloads/co.sql", "r") 
# file = open("out.txt", "r") 

prefix = "/Users/petergrabowski/Desktop/raw_files/"
files = ["co.sql", "s4s.sql", "po.sql", "k.sql", "tg.sql", "g.sql", "vg.sql", "a.sql"]

i = 0

window = 100

def is_comment(x):

	if "NULL" in x:
		return False
	if len(x) < 4:
		return False
	if "anonymous" in x.lower():
		return False
	if len(x.split(" ")) < 2:
		return False
	if ".jpg" in x:
		return False
	if ".png" in x:
		return False
	try:
		if x[0] == '(':
			xint = int(x[1:])
			return False
		else:
			xint = int(x)
			return False
	except:
		pass

	return True

def remove_comment_number(x):
	return re.sub(">>\d+", "", x)

def clean_up_tags(x):
	return x.replace("[spoiler]","").replace("[/spoiler]","")

for filename in files:
	already_created = False
	full_name = prefix + filename
	board_name = filename.replace(".sql","")
	out_name = "/Users/petergrabowski/Desktop/out_files/4chan_%s.txt" % board_name
	print full_name

	if os.path.isfile(out_name):
		print "already created"
		pass
	if not already_created:
		with open(full_name, "r") as infile, open(out_name, 'w') as outfile:
		# with open("4chan.txt", 'w') as outfile:
				for line in infile:
					content = line[line.find("(")+1:line.rfind(")")]

					f = StringIO(content)
					csv_reader = reader(f, delimiter=",", quotechar="\"")
					for xline in csv_reader:
						for x in xline:
							if is_comment(x):
								no_comments = remove_comment_number(x)
								no_tags = clean_up_tags(no_comments)
								clean_spacing = no_tags.replace("\\\\","").replace("\\n","").replace("\\","")
								outfile.write(clean_spacing + "\n")
