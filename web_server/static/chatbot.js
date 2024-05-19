// create a socket with the server in order to communicate
var socket = io.connect('ws://' + document.domain + ':' + location.port);

var chatBox = document.getElementById('chat-box');
var userInput = document.getElementById('user-input');
var sendButton = document.getElementById('send-button');
var stopButton = document.getElementById('stop-button');
var tempInput = document.getElementById('temp')
var topPInput = document.getElementById('top_p')

isGenerating = false;
currentResponseContainer = null;

function start_generating() {
    /**
    Called when the user click the send button. Checks that the input is valid and sends the prompt to the server. Creates a new message in the chatbox for the user's question and the chatbot's response.
    */
    var input = userInput.value;
    var temperature = tempInput.value;
    var top_p_val = topPInput.value;

    if (input.trim() == '')
        return;

    userInput.value = '';

    onGenerationStart();

    // create new messages, one for the user's question and the other for the chatbot's response
    addMessage('user', input);
    currentResponseContainer = addMessage('chatbot', '')

    // send the prompt and inference config to the server
    socket.emit('generate_chatbot_response', {prompt: input, temp: temperature, top_p: top_p_val});
};

function stop_generating() {
    /**
    Called when the user clicks the stop button. Tells the server to stop generating a response and allows the user to send a new question.
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
    if (data['token'] == '</s>' || currentResponseContainer == null) {
        onGenerationEnd();
        return;
    }

    // add the new token to the current chatbot message
    currentResponseContainer.innerHTML += data['token'];
});


function addMessage(sender, message) {
    /**
    Renders a new message in the chatbox. This function is used both for user messages and chatbot messages.
    */
    var messageElement = document.createElement('div');

    // add the message element's classes, the sender class is used to ascertain whether its a user message or a chatbot message
    messageElement.classList.add('message', sender);

    var senderSpan = document.createElement('span');
    senderSpan.className = 'message-sender';
    senderSpan.innerHTML = sender;
    messageElement.appendChild(senderSpan);

    messageElement.appendChild(document.createElement('br'));

    var contentSpan = document.createElement('span');
    contentSpan.className = 'message-content';
    contentSpan.innerHTML = message;
    messageElement.appendChild(contentSpan);

    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;

    return contentSpan;
}

function onGenerationStart() {
    /**
    Updates the send/stop buttons when the chatbot starts generating a response.
    */
    isGenerating = true;

    stopButton.disabled = false;
    stopButton.style.background = '#007bff';
    sendButton.disabled = true;
}

function onGenerationEnd() {
    /**
    Updates the send/stop buttons when the chatbot stops generating a response.
    */
    isGenerating = false;

    stopButton.disabled = true;
    stopButton.style.background = 'grey';
    sendButton.disabled = false;
}