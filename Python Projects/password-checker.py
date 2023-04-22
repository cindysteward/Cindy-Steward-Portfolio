#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:21:56 2022

@author: cindysteward
"""

import tkinter
    
def greenish():
    window.configure(background="#7CFC00")

def reddish():
    window.configure(background="#FF0000")
    
def getPass():
        global secret_word
        secret_word = ent1.get()
        if secret_word == 'hoompa':
            greenish()
        else:
            reddish()
        
# Define a new tkinter window, with some layout choices.
window = tkinter.Tk()
window.configure(background="#a1dbcd")
window.geometry("300x300")
window.title('Password Checker')
window.attributes('-topmost', True) # note - before topmost

something = []
#create a label widget called 'type something'
lbl1 = tkinter.Label(window, text="Type Password",bg="#a1dbcd")


#create a text entry widget called 'ent1'
ent1 = tkinter.Entry(window)

# button that changes the background color, by calling the function 'pnk' defined above
pnkbtn = tkinter.Button(window, text="Check", command=getPass)    

#(pady is padding in the y direction (above and below), this creates some space)
lbl1.pack(pady=5)
ent1.pack(pady=5)
pnkbtn.pack(pady=5)

# NOTE the line below is needed in most cases, but not if Tkinter is already set 
# as the graphics back end in the preferences.
window.mainloop()
