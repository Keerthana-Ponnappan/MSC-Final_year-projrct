function start() {
    let res_msg = document.createElement('div');
    res_msg.innerHTML = "Hey I'm silly bot, are you looking for something, I'm here to help you !!";
    res_msg.setAttribute("class", "left");

    document.getElementById('msg-area').appendChild(res_msg);
}

function togglediv(id) {
    var div = document.getElementById(id);
    div.style.display = div.style.display == "none" ? "block" : "none";
}

function send() {
    document.getElementById('send').addEventListener("click", async(e) => {
        e.preventDefault();
        var req = document.getElementById('text').value;
        if (req == undefined || req == "") {

        } else {
            let res = document.getElementById('text').value;
            //await axios.get('hhtps://api.monkedev.com/fun/chat?msg=${req}').then(data => {
            //    res = JSON.stringify(data.data.response)
            //});

            let msg_req = document.createElement('div');
            let msg_res = document.createElement('div');

            let Con1 = document.createElement('div');
            let Con2 = document.createElement('div');

            Con1.setAttribute("class", "Con1");
            Con2.setAttribute("class", "Con2");

            msg_req.innerHTML = req;
            msg_res.innerHTML = res;

            msg_req.setAttribute("class", "right");
            msg_res.setAttribute("class", "left");

            let message = document.getElementById('msg-area');

            message.appendChild(Con1);
            message.appendChild(Con2);

            Con1.appendChild(msg_req); //user
            Con2.appendChild(msg_res); //bot

            document.getElementById('text').value = "";



        }

        function scroll() {
            var scrollmsg = document.getElementById('msg-area');
            scrollmsg.scrollTop = scrollmsg.scrollHeight;
        }

        scroll();



    });


}