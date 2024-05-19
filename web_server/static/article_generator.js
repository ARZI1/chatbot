// create a socket with the server in order to communicate
var socket = io.connect('ws://' + document.domain + ':' + location.port);

var textArea = document.getElementById('user-input');
var sendButton = document.getElementById('send-button');
var stopButton = document.getElementById('stop-button');
var tempInput = document.getElementById('temp')
var topPInput = document.getElementById('top_p')


isGenerating = false;

function start_generating() {
    /**
    Called when the send button is pressed. Sends the server the prompt and inference settings.
    */
    var input = textArea.value;
    var temperature = tempInput.value;
    var top_p_val = topPInput.value;

    onGenerationStart();

    socket.emit('generate_article', {prompt: input, temp: temperature, top_p: top_p_val});
}

function stop_generating() {
    /**
    Called when the stop button is pressed, tells the server to stop generating a response.
    */
    if (!isGenerating)
        return;

    onGenerationEnd();
    socket.emit('stop_generating', {user_id: 'dang'});
}

socket.on('response', function(data) {
    /**
    Handles responses from the server. Every token generated and sent by the server will call this event.
    */
    if (data['token'] == '</s>') {
        onGenerationEnd();
        return;
    }

    // add the new token to the text box
    textArea.value += data['token'];
});

function onGenerationStart() {
    /**
    Updates the send/stop buttons when the model start generating a response.
    */
    isGenerating = true;

    stopButton.disabled = false;
    stopButton.style.background = '#007bff';
    sendButton.disabled = true;
}

function onGenerationEnd() {
    /**
    Updates the send/stop buttons when the model stops generating a response.
    */
    isGenerating = false;

    stopButton.disabled = true;
    stopButton.style.background = 'grey';
    sendButton.disabled = false;
}