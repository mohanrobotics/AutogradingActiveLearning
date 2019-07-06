var express = require('express');
var app = express();
var path = require('path');

// viewed at http://localhost:8080
app.get('/', function(req, res) {
    res.sendFile(path.join(__dirname + '/index.html'));
});

app.get('/newhtml', function(req, res) {
    res.sendFile(path.join(__dirname + '/new_index.html'));
});

app.use(express.static('public'));
app.listen(8080);
