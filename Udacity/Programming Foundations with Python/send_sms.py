# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 14:06:53 2018

@author: HP
"""

from twilio.rest import Client

# SID da sua conta, encontre em twilio.com/console
account_sid = "AC25d8f1869100730e156c802491463c1f"
# Seu Auth Token, encontre em twilio.com/console
auth_token  = "88279c88f032f2e510ca50ab59d6967f"

client = Client(account_sid, auth_token)

message = client.messages.create(
    to="+5511986820342", 
    from_="+16502854887",
    body="Booooo!")

print(message.sid)