# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import urllib2
import zlib
import re
import nltk.data


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
file = open("book_indexes.txt", "r") 

def generate_gutenberg_utf_url(index, rare=False):
	if not rare:
		url = "http://www.gutenberg.org/cache/epub/%d/pg%d.txt" % (index, index)
		print url
	else:
		url = "http://www.gutenberg.org/files/%d/%d-0.txt" % (index, index)
		print url
	return url

def get_raw_book(index):
	url = generate_gutenberg_utf_url(index)

	try:
		page = urllib2.urlopen(url)
	except:
		url = generate_gutenberg_utf_url(index, rare=True)
		page = urllib2.urlopen(url)


	contents = page.read()

	# check if compressed
	if "\x1f\x8b" in contents:
		contents = zlib.decompress(bytes(contents), 15+32)

	return contents

def check_start(line):
	lower_line = line.lower()
	if "start of this project gutenberg" in lower_line:
		return True
	if "start of this project gutenberg ebook" in lower_line:
		return True
	if "START OF THE PROJECT GUTENBERG EBOOK" in line:
		return True
	return False

def check_end(line):
	lower_line = line.lower()
	if "end of project gutenberg" in lower_line:
		return True
	if "end of the project gutenberg" in lower_line:
		return True
	return False

def strip_book_metadata(raw_book):
	keep_line = False
	book_data = []
	for line in raw_book.splitlines():
		if check_end(line):
			keep_line = False
		if keep_line and line != "":
			book_data.append(line.rstrip())
		if check_start(line):
			keep_line = True
	return book_data

def fix_formatting(book):
	entire_book = " ".join(book)
	no_double_spaces = entire_book.replace("  "," ")

	# print no_double_spaces

	book_sents = tokenizer.tokenize(no_double_spaces)
	return book_sents

def write_book_to_file(index, book_data, path="./"):
	filename = "%s%d.txt" % (path,index)
	print filename
	with open(filename, "w") as file:
		file.write("\n".join(book_data))


sample_file = [ 10360]
bad_books = []
# for line in sample_file: 
for line in file: 
	try:
		index = int(line)

		raw_book = get_raw_book(index)
		book_data = strip_book_metadata(raw_book)
		if book_data:
			formatted_book = fix_formatting(book_data)
			write_book_to_file(index, formatted_book, "./books/")
		else:
			bad_books.append(line)
		print "\n"
		# print book_data
	except:
		bad_books.append(line)
		print "\n"

print "bad books", bad_books
