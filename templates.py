# templates.py
HTML_TEMPLATES = {
    "bot": '''
    <div class="chat-message bot">
        <div class="avatar">
            <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
        </div>
        <div class="message">{{MSG}}</div>
    </div>
    ''',
    "user": '''
    <div class="chat-message user">
        <div class="avatar">
            <img src="https://i.postimg.cc/gcM4CjzT/Pavan.jpg">
        </div>    
        <div class="message">{{MSG}}</div>
    </div>
    '''
}
