# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:34:20 2016

@author: finn
"""

class Animal(object):
    label = "Animal"
    def init(self):
        print self.noise()
        
    def noise(self):
        return "hmm"
        
class Dog(Animal):
    label = "Dog"
    def noise(self):
        return "woof"