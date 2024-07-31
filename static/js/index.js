let chat_messages = document.get

document.onload = () =>{
    let messages

    fetch("../../chats/messages.json")
        .then(response => response.json())
        .then(data => {
            messages = messages
        })
    
}




