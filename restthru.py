#   RestThru is a mobile robotics framework developed at the
#   School of Electrical and Computer Engineering, University
#   of Campinas, Brazil by Eleri Cardozo and collaborators.
#   eleri@dca.fee.unicamp.br
#
#   Copyright (C) 2018 Eleri Cardozo
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import http.client
import json
from urllib.parse import urlparse


# INIT - set server address ([session ID])
def http_init(*sessid):
	global sessionId
	if len(sessid) > 0:
		sessionId = sessid[0]
	else:
		sessionId = ''


# GET - return content as dict containing JSON body
def http_get(resource):
	url = urlparse(resource)
	if url.scheme == 'http':
		conn = http.client.HTTPConnection(url.netloc)
	elif url.scheme == 'https':
		conn = http.client.HTTPSConnection(url.netloc)
	else: return ['', 400]
	resource = url.path
	if url.query != '': resource += '?' + url.query
	headers = {}
	headers['Accept'] = "text/plain,application/json"
	if len(sessionId) > 0:
		headers['Cookie'] = sessionId + "=http://" + serverAddress + resource 
	conn.request("GET", resource, None, headers)
	resp = conn.getresponse()
	key = ''
	retdata = ''
	jdict = {}
	if resp.status == 200:
		jbytes = resp.read()
		retdata = jbytes.decode('ASCII')
		ctype = resp.getheader('Content-Type')
		if ctype.startswith('application/json'):
			retdata = json.loads(retdata)
			bres = resource.rfind("/")
			eres = resource.find("?")
			if bres >= 0 and eres > 0:
				key = resource[bres+1:eres]
			elif bres >= 0:
				key = resource[bres+1:]
			if key != '':
				retdata = retdata[key]
	conn.close()
	return [retdata, resp.status]


# POST - return body or 'Location' header if status is 201 Created
def http_post(resource, payload):
	url = urlparse(resource)
	if url.scheme == 'http':
		conn = http.client.HTTPConnection(url.netloc)
	elif url.scheme == 'https':
		conn = http.client.HTTPSConnection(url.netloc)
	else: return ['', 400]
	resource = url.path
	if url.query != '': resource += '?' + url.query
	key = ''
	retdata = ''
	bres = resource.rfind("/")
	eres = resource.find("?")
	if bres >= 0 and eres > 0:
		key = resource[bres+1:eres]
	elif bres >= 0:
		key = resource[bres+1:]
	if key != '':
		resstate = {}
		resstate[key] = payload
	else:
		resstate = payload
	headers = {}
	headers['Content-Type'] = "application/json"
	headers['Accept'] = "text/plain,application/json"
	if len(sessionId) > 0:
		headers['Cookie'] = sessionId + "=http://" + serverAddress + resource 
	conn.request("POST", resource, json.dumps(resstate), headers)
	resp = conn.getresponse()
	if resp.status == 200 or resp.status == 202:   # OK or Accepted
		rbytes = resp.read()
		retdata = rbytes.decode('ASCII')
		ctype = resp.getheader('Content-Type')
		if ctype.startswith('application/json'):
			retdata = json.loads(retdata)
			if(key != ''):
				retdata = retdata[key]
	elif resp.status == 201:   # Created
		retdata = resp.getheader('Location')
	conn.close()
	return [retdata, resp.status]


# PUT - return body if 200 OK or failure reason 
def http_put(resource, payload):
	url = urlparse(resource)
	if url.scheme == 'http':
		conn = http.client.HTTPConnection(url.netloc)
	elif url.scheme == 'https':
		conn = http.client.HTTPSConnection(url.netloc)
	else: return ['', 400]
	resource = url.path
	if url.query != '': resource += '?' + url.query
	key = ''
	retdata = ''
	bres = resource.rfind("/")
	eres = resource.find("?")
	if bres >= 0 and eres > 0:
		key = resource[bres+1:eres]
	elif bres >= 0:
		key = resource[bres+1:]
	if key != '':
		resstate = {}
		resstate[key] = payload
	else:
		resstate = payload
	headers = {}
	headers['Content-Type'] = "application/json"
	headers['Accept'] = "text/plain,application/json"
	if len(sessionId) > 0:
		headers['Cookie'] = sessionId + "=http://" + serverAddress + resource 
	conn.request("PUT", resource, json.dumps(resstate), headers)
	resp = conn.getresponse()
	if resp.status == 200 or resp.status == 202:   # OK or Accepted
		rbytes = resp.read()
		retdata = rbytes.decode('ASCII')
		ctype = resp.getheader('Content-Type')
		if ctype.startswith('application/json'):
			retdata = json.loads(retdata)
			if(key != ''):
				retdata = retdata[key]
	conn.close()
	return [retdata, resp.status]


# DELETE - return status code
def http_delete(resource):
	url = urlparse(resource)
	if url.scheme == 'http':
		conn = http.client.HTTPConnection(url.netloc)
	elif url.scheme == 'https':
		conn = http.client.HTTPSConnection(url.netloc)
	else: return [False, 400]
	resource = url.path
	if url.query != '': resource += '?' + url.query
	retdata = False
	headers = {}
	if len(sessionId) > 0:
		headers['Cookie'] = sessionId + "=http://" + serverAddress + resource 
	conn.request("DELETE", resource, None, headers)
	resp = conn.getresponse()
	conn.close()
	if resp.status == 200: 
		retdata = True
	return [retdata, resp.status]

