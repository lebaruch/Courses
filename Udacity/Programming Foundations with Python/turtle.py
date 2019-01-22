# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import turtle

#Function to loop 4 times the forward and right functions
def draw_square(some_turtle):
    for i in range(1,5):
        some_turtle.forward(100)
        some_turtle.right(90)


def draw_forms():
    window = turtle.Screen()
    window.bgcolor("black")
    #Draw Jake the square
    jake = turtle.Turtle()
    jake.shape("classic")
    jake.color("orange")
    jake.speed(10)
    for i in range(1,36):
        draw_square(jake)
        jake.right(10)
 #Draw bolota the circle
    #bolota = turtle.Turtle()
    #bolota.shape("turtle")
    #bolota.color("blue")
    #bolota.circle(80)
 #Draw trolota the triangle
    #trolota = turtle.Turtle()
    #trolota.shape("classic")
    #trolota.color("white")
    #trolota.forward(100)
    #trolota.left(120)
    #trolota.forward(100)
    #trolota.left(120)
    #trolota.forward(100)
    
    window.exitonclick()
    
draw_forms()