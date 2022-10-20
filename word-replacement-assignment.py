#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:26:36 2022

@author: cindysteward
"""
#%%
#Replace Word Assignment: 
#Write a function called ‘sed’ that takes as its 4 arguments; 1) a pattern string, 2) a replacement string, and 3 & 4; two filenames (also strings); it should read/open the first file and write the contents into the second file (creating it if necessary).
#If the pattern string appears anywhere in the source file, it should be replaced with the replacement string in the new file.
#If an error occurs while opening, reading, writing or closing files, your program should catch the exception, print an error message, and stop running.

#%%

import sys # the sys module is imported to support the exit command 

def sed (pattern, replace, sourcefile, newfile):
    try: # asking computer to try the following code until encoutering the exception 
        with open (sourcefile, "r") as f1: #allows code to open the file bob.txt and read it
            for lines in f1: # for loop: for the content in the file, 
                if pattern in lines: # if the pattern string is in the content, 
                    #BUG: changed line into lines in line 7 and line 10: so it lookes in the "lines" ALSO used in the for-loop.
                    with open (newfile, "w") as f2: # open a new file, give it a name, and
                        f2.write(lines.replace(pattern, replace)) # copy the strings from the source file while replacing the pattern string with the replacement string
                    f2.close()  # close the created file
                else: # runs if the given pattern string doesn't exist in the source file
                    print("Cannot find the given word in file...")
                    sed(str(input ("Which word would you like to replace? ")), str(input ("To be replaced by the word: ")), input("File to look in: ") + ".txt", input("New file name: ") + ".txt")
                    #BUG: see the change as at the end of the file (at the function call), about the input file.
        f1.close() 
        
        print("The new file has been saved to your computer.") #BUG: instead of always being printed AFTER function call, it is now in the function and part of the function. This line was before debugging placed under the function call.
        
    except OSError: # this is an exception for a system error where the function cannot open/read the source file 
        print("ERROR: CANNOT USE GIVEN SOURCE FILE")
        sys.exit() # allows the function to stop running when exception is encountered
        

sed(str(input ("Which word would you like to replace? ")), str(input ("To be replaced by the word: ")), input("File to look in: ") + ".txt", input("New file name: ") + ".txt")
#BUG: the input file, or the file that was going to be looked in, was written as just "bob.txt", which always caused an OSError. I rewrote it in similar fashion as the other input files, so now you can fill one in.         