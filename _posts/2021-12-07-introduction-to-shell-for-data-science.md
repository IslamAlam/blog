---
title: Introduction to Shell for Data Science
date: 2021-12-07 11:22:07 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Introduction to Shell for Data Science
=========================================
  






 This is the memo of the 19th course of ‘Data Scientist with Python’ track.
   

**You can find the original course
 [HERE](https://www.datacamp.com/courses/introduction-to-shell-for-data-science)**
 .
 




---



# **1. Manipulating files and directories**
------------------------------------------


#### 
**How does the shell compare to a desktop interface?**



 They are both interfaces for issuing commands to the operating system.
 


#### 
**Some Basic commands: pwd, ls**



 pwd —
 **p** 
 rint
 **w** 
 orking
 **d** 
 irectory
   

 ls —
 **l** 
 i
 **s** 
 ting
 


#### 
**Absolute Path vs. Relative Path**



 The shell decides if a path is absolute or relative by looking at its first character: if it begins with
 `/` 
 , it is absolute, and if it doesn’t, it is relative.
 




#### 
**Some Basic commands: cd**



 cd — change directory
 


#### 
**Directory Basic**



 . — the current directory
   

 .. — the directory above the one I’m currently in
   

 ~ — your home directory
 


#### 
**Some Basic commands: cp, mv**



 cp — copy
   

 cp fileA fileB # copy fileA to fileB
   

 cp fileA fileB fileC NewDirectory #copy fileA fileB fileC to NewDirectory
   

  

 mv — move (when a new destination directory is specified)
   

 mv – rename (when no new destination directory is specified)
   

 same syntax as copy
   

 mv fileA fileB # rename fileA to fileB
   

 mv fileA fileB fileC NewDirectory #move fileA fileB fileC to NewDirectory
   

  

**# caution: both cp and mv will overwrite the existing file** 



#### 
**Some Basic commands: rm, rmdir, mkdir**



 rm — remove files
   

 rm fileA fileB
   

**# caution: if a file is removed, it’s removed forever** 
  

  

 rmdir — remove an empty directory
   

 rmdir directoryA
   

  

 mkdir — make a directory
   

 mkdir directoryA
 




---



# **2. Manipulating data**
-------------------------


#### 
**Some Basic commands: cat**



 cat — concatenate, show file contents
   

 cat fileA
 


#### 
**Some Basic commands: less**



 less — is more, show file contents
   

 less fileA fileB
   

  

 when less fileA
   

 space — page down
   

`:n` 
 — to move to the next file
   

`:p` 
 — to go back to the previous one
   

`:q` 
 — to quit.
 


#### 
**Some Basic commands: head**



 head — display first 10 lines of a file
   

 head fileA
 


#### 
**Tips: tab completion**



 If you start typing the name of a file and then press the tab key, the shell will do its best to auto-complete the path.
 


#### 
**command-line flag**



 head -n 3 fileA # display first n(n = 3 here) lines of a file
   

 -n —
 **n** 
 umber of lines
 


#### 
**ls -RF**



 ls -RF directoryA
   

 ls -FR directoryA
   

 ls -F -R directoryA
   

 ls -R -F directoryA
   

 are all the same command
   

  

 -R — recursive, list everything below a directory
   

 -F — prints a / after the name of every directory and a * after the name of every runnable program.
 


#### 
**Some Basic commands: man**



 man — manual
   

 man head
 



![Desktop View]({{ site.baseurl }}/assets/datacamp/introduction-to-shell-for-data-science/capture-20.png)


`man` 
 automatically invokes
 `less` 
 , so you may need to press spacebar to page through the information and
 `:q` 
 to quit.
 



 The one-line description under
 `NAME` 
 tells you briefly what the command does, and the summary under
 `SYNOPSIS` 
 lists all the flags it understands. Anything that is optional is shown in square brackets
 `[...]` 
 , either/or alternatives are separated by
 `|` 
 , and things that can be repeated are shown by
 `...` 
 , so
 `head` 
 ‘s manual page is telling you that you can
 *either* 
 give a line count with
 `-n` 
 or a byte count with
 `-c` 
 , and that you can give it any number of filenames.
 


#### 
**Some Basic commands: tail**



 tail — display last 10 lines of a file
   

 tail -n 1 fileA # display last 1 lines of a fileA
   

 tail -n +2 fileA # display from the 2nd lines to the end of a fileA
 


#### 
**Some Basic commands: cut**



 cut -f 2-5,8 -d , fileA.csv
   

 -f 2-5, 8 — fields, select columns 2 through 5 and columns 8, using comma as the separator
   

 -d , — delimiter, use ‘,’ as delimiter
   

  

 cut -f2 -d , fileA.csv
   

 cut -f 2 -d , fileA.csv
   

 are the same.
   

 Space is optional between -f and 2
 


#### 
**repeat commands**



 history — print a list of commands you have run recently
   

 !some_command — run the last some_command again(ex. !cut)
   

 !2 — run the 2nd command listed in history
 


#### 
**Some Basic commands: grep**



 grep patternA fileA
 


* `-c` 
 : print a count of matching lines rather than the lines themselves
* `-h` 
 : do
 *not* 
 print the names of files when searching multiple files
* `-i` 
 : ignore case (e.g., treat “Regression” and “regression” as matches)
* `-l` 
 : print the names of files that contain matches, not the matches
* `-n` 
 : print line numbers for matching lines
* `-v` 
 : invert the match, i.e., only show lines that
 *don’t* 
 match




```

$ grep molar seasonal/autumn.csv
2017-02-01,molar
2017-05-25,molar

$ grep -nv molar seasonal/spring.csv
1:Date,Tooth
2:2017-01-25,wisdom
3:2017-02-19,canine
...
8:2017-03-14,incisor
10:2017-04-29,wisdom
11:2017-05-08,canine
...
22:2017-08-13,incisor
23:2017-08-13,wisdom

$ grep -c incisor seasonal/autumn.csv seasonal/winter.csv
seasonal/autumn.csv:3
seasonal/winter.csv:6

```


#### 
**Some Basic commands: paste**



 paste — to combine data files
 




```

$ paste -d , seasonal/autumn.csv seasonal/winter.csv
Date,Tooth,Date,Tooth2017-01-05,canine,2017-01-03,bicuspid
2017-01-17,wisdom,2017-01-05,incisor
2017-01-18,canine,2017-01-21,wisdom
...
2017-08-16,canine,2017-07-01,incisor
,2017-07-17,canine
,2017-08-10,incisor
...

# The last few rows have the wrong number of columns.

```




---



# **3. Combining tools**
-----------------------


#### 
**Store a command’s output in a file**



 some_command > new_file
   

 ex. tail -n 5 seasonal/winter.csv > last.csv
 


#### 
**combine commands**



 command_A | command_B | …
 




```

$ cut -f 2 -d , seasonal/summer.csv | grep -v Tooth
canine
wisdom
bicuspid
...

```


#### 
**Some Basic commands: wc**



 wc — word count, prints the number of characters, words, and lines in a file. You can make it print only one of these using
 `-c` 
 ,
 `-w` 
 , or
 `-l` 
 respectively.
   

  

 $ grep 2017-07 seasonal/spring.csv | wc -l
 


#### 
**wildcards: ***



`*` 
 , which means “match zero or more characters”.
 




```

$ head -n 3 seasonal/s*.csv
==> seasonal/spring.csv <==
Date,Tooth
2017-01-25,wisdom
2017-02-19,canine

==> seasonal/summer.csv <==
Date,Tooth
2017-01-11,canine
2017-01-18,wisdom

```


#### 
**wildcards: ?, [], {}**


* `?` 
 matches a single character, so
 `201?.txt` 
 will match
 `2017.txt` 
 or
 `2018.txt` 
 , but not
 `2017-01.txt` 
 .
* `[...]` 
 matches any one of the characters inside the square brackets, so
 `201[78].txt` 
 matches
 `2017.txt` 
 or
 `2018.txt` 
 , but not
 `2016.txt` 
 .
* `{...}` 
 matches any of the comma-separated patterns inside the curly brackets, so
 `{*.txt, *.csv}` 
 matches any file whose name ends with
 `.txt` 
 or
 `.csv` 
 , but not files whose names end with
 `.pdf` 
 .


#### 
**Some Basic commands: sort**



 sort — By default it does this in ascending alphabetical order
   

`-n` 
 and
 `-r` 
 can be used to sort numerically and reverse the order of its output
   

`-b` 
 tells it to ignore leading blanks
   

`-f` 
 tells it to
 **f** 
 old case (i.e., be case-insensitive)
 




```

$ sort -r  seasonal/summer.csv
Date,Tooth
2017-08-04,canine
2017-08-03,bicuspid
2017-08-02,canine
...

```


#### 
**Some Basic commands: uniq**



 uniq — remove duplicated lines
 



 If a file contains:
 




```

2017-07-03
2017-07-03
2017-08-03
2017-08-03

```



 then
 `uniq` 
 will produce:
 




```

2017-07-03
2017-08-03

```



 but if it contains:
 




```

2017-07-03
2017-08-03
2017-07-03
2017-08-03

```



 then
 `uniq` 
 will print all four lines.
 



 The reason is that
 `uniq` 
 is built to work with very large files. In order to remove non-adjacent lines from a file, it would have to keep the whole file in memory (or at least, all the unique lines seen so far). By only removing adjacent duplicates, it only has to keep the most recent unique line in memory.
 


* get the second column from
 `seasonal/winter.csv` 
 ,
* remove the word “Tooth” from the output so that only tooth names are displayed
* sort the output so that all occurrences of a particular tooth name are adjacent
* display each tooth name once along with a count of how often it occurs




```

$ cut -d , -f 2 seasonal/winter.csv | grep -v Tooth | sort | uniq -c
      4 bicuspid
      7 canine
      6 incisor
      4 molar
      4 wisdom

```


#### 
**stop a running program**



`Ctrl` 
 +
 `C` 



#### 
**Wrapping up**




```

$ wc -l seasonal/*.csv  21 seasonal/autumn.csv
  24 seasonal/spring.csv  25 seasonal/summer.csv
  26 seasonal/winter.csv
  96 total

$ wc -l seasonal/*.csv | grep -v total
  21 seasonal/autumn.csv
  24 seasonal/spring.csv
  25 seasonal/summer.csv
  26 seasonal/winter.csv

$ wc -l seasonal/*.csv | grep -v total | sort -n | head -n 1
  21 seasonal/autumn.csv

```




---



# **4. Batch processing**
------------------------


#### 
**environment variables**





| 
 Variable
  | 
 Purpose
  | 
 Value
  |
| --- | --- | --- |
| `HOME`  | 
 User’s home directory
  | `/home/repl`  |
| `PWD`  | 
 Present working directory
  | 
 Same as
 `pwd` 
 command
  |
| `SHELL`  | 
 Which shell program is being used
  | `/bin/bash`  |
| `USER`  | 
 User’s ID
  | `repl`  |




 To get a complete list (which is quite long), you can type
 `set` 
 in the shell.
   

  

 ex. HISTFILESIZE determines how many old commands are stored in your command history.
 




```

$ set | grep HISTFILESIZE
HISTFILESIZE=2000

```


#### 
**print variable
 
 echo $variable_name**



 echo — prints its arguments.
   

  

 To get the variable’s value, you must put a dollar sign
 `$` 
 in front of it.
   

  

 This is true everywhere: to get the value of a variable called
 `X` 
 , you must write
 `$X` 
 . (This is so that the shell can tell whether you mean “a file named X” or “the value of a variable named X”.)
 




```

$ echo $OSTYPE
linux-gnu

```


#### 
**shell variable
 
 variable_name=value**



 To create a shell variable, you simply assign a value to a name
 *without* 
 any spaces before or after the
 `=` 
 sign.
 




```

$ testing=seasonal/winter.csv
$ head -n 1 $testing
Date,Tooth

```


#### 
**loops**




```

for filetype in gif jpg png; do echo $filetype; done

```



 it produces:
 




```

gif
jpg
png

```



 Notice these things about the loop:
 


1. **The structure is
 `for` 
 …variable…
 `in` 
 …list…
 `; do` 
 …body…
 `; done`**
2. The list of things the loop is to process (in our case, the words
 `gif` 
 ,
 `jpg` 
 , and
 `png` 
 ).
3. The variable that keeps track of which thing the loop is currently processing (in our case,
 `filetype` 
 ).
4. The body of the loop that does the processing (in our case,
 `echo $filetype` 
 ).



 Notice that the body uses
 `$filetype` 
 to get the variable’s value instead of just
 `filetype` 
 , just like it does with any other shell variable. Also notice where the semi-colons go: the first one comes between the list and the keyword
 `do` 
 , and the second comes between the body and the keyword
 `done` 
 .
 


#### 
**loops with wildcard ***




```

$ for filename in seasonal/*.csv; do echo $filename; doneseasonal/autumn.csv
seasonal/spring.csv
seasonal/summer.csv
seasonal/winter.csv

```


#### 
**loops with variable $**




```

$ files=seasonal/*.csv
$ for f in $files; do echo $f; done
seasonal/autumn.csv
seasonal/spring.csv
seasonal/summer.csv
seasonal/winter.csv

```


#### 
**loops with pipe |**




```

$ for file in seasonal/*.csv; do head -n 2 $file | tail -n 1; done
2017-01-05,canine
2017-01-25,wisdom
2017-01-11,canine
2017-01-03,bicuspid

$ for file in seasonal/*.csv; do grep -h 2017-07 $file; done
2017-07-10,incisor
2017-07-10,wisdom
2017-07-20,incisor
...

```


#### 
**Avoiding use space in file_name**



 use ‘ or ” if there is a space in file_name
 




```

mv 'July 2017.csv' '2017 July data.csv'

```


#### 
**loops with several commands**



 seperate commands with
 **;** 





```

$ for f in seasonal/*.csv; do echo $f head -n 2 $f | tail -n 1; done
seasonal/autumn.csv head -n 2 seasonal/autumn.csv
seasonal/spring.csv head -n 2 seasonal/spring.csv
seasonal/summer.csv head -n 2 seasonal/summer.csv
seasonal/winter.csv head -n 2 seasonal/winter.csv

$ for f in seasonal/*.csv; do echo $f; head -n 2 $f | tail -n 1; done
seasonal/autumn.csv
2017-01-05,canine
seasonal/spring.csv
2017-01-25,wisdom
seasonal/summer.csv
2017-01-11,canine
seasonal/winter.csv
2017-01-03,bicuspid

```




---



# **5. Creating new tools**
--------------------------


#### 
**Edit file with nano**



 Unix has a bewildering variety of text editors. For this course, we will use a simple one called Nano. If you type
 `nano filename` 
 , it will open
 `filename` 
 for editing (or create it if it doesn’t already exist). You can move around with the arrow keys, delete characters using backspace, and do other operations with control-key combinations:
 


* `Ctrl` 
 +
 `K` 
 : delete a line.
* `Ctrl` 
 +
 `U` 
 : un-delete a line.
* `Ctrl` 
 +
 `O` 
 : save the file (‘O’ stands for ‘output’).
* `Ctrl` 
 +
 `X` 
 : exit the editor.


#### 
**Save history commands for future use**




```

$ cp seasonal/s*.csv ~/
$ grep -hv Tooth s*.csv > temp.csv
$ history | tail -n 3 > steps.txt
$ cat steps.txt
    9  cp seasonal/s*.csv ~/
   10  grep -hv Tooth s*.csv > temp.csv
   11  history | tail -n 3 > steps.txt

```


#### 
**run shell script**



 run shell script using
   

 bash script.sh
 




```

$ nano dates.sh
$ cat dates.sh
cut -d , -f 1 seasonal/*.csv
$ bash dates.sh
Date
2017-01-05
2017-01-17
2017-01-18
...

```


#### 
**save a script output into a file**




```

$ nano teeth.sh
$ cat teeth.sh
cut -d , -f 2 seasonal/*.csv | grep -v Tooth | sort | uniq -c
$ bash teeth.sh > teeth.out
$ cat teeth.out
     15 bicuspid
     31 canine
     18 incisor
     11 molar
     17 wisdom

```


#### 
**pass filenames to scripts $@**



 if
 `unique-lines.sh` 
 contains this:
 




```

sort $@ | uniq

```



 then when you run:
 




```

bash unique-lines.sh seasonal/summer.csv

```



 the shell replaces
 `$@` 
 with
 `seasonal/summer.csv` 
 and processes one file. If you run this:
 




```

bash unique-lines.sh seasonal/summer.csv seasonal/autumn.csv

```



 it processes two data files, and so on.
 




```

$ nano count-records.sh
$ cat count-records.sh
tail -q -n +2 $@ | wc -l

$ bash count-records.sh seasonal/*.csv > num-records.out
$ head num-records.out
92

```


#### 
**command-line parameters**



 As well as
 `$@` 
 , the shell lets you use
 `$1` 
 ,
 `$2` 
 , and so on to refer to specific command-line parameters.
 



 The script
 `get-field.sh` 
 is supposed to take a filename, the number of the row to select, the number of the column to select, and print just that field from a CSV file.
   

  

 bash get-field.sh seasonal/summer.csv 4 2
   

 should select the second field from line 4 of
 `seasonal/summer.csv` 
 .
 




```

head -n $2 $1 | tail -n 1 | cut -d , -f $3
bash get-field.sh seasonal/summer.csv 4 2

```


#### 
**scripts with 2 or more lines**




```

$ cat range.sh
wc -l $@ | grep -v total | sort -n | head -n 1
wc -l $@ | grep -v total | sort -nr | head -n 1
$ bash range.sh seasonal/*.csv > range.out
$ head range.out
  21 seasonal/autumn.csv
  26 seasonal/winter.csv

```


#### 
**scripts with loops**




```

$ cat date-range.sh
# Print the first and last date from each data file.
for filename in $@
do
    cut -d , -f 1 $filename | grep -v Date | sort | head -n 1
    cut -d , -f 1 $filename | grep -v Date | sort | tail -n 1
done

$ bash date-range.sh seasonal/*.csv
2017-01-05
2017-08-16
2017-01-25
...

$ bash date-range.sh seasonal/*.csv | sort
2017-01-03
2017-01-05
2017-01-11
...

```



 The End.
   

 Thank you for reading.
 


