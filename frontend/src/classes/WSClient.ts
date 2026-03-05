'use strict';

import {
    Tools,
    Emitter,
    ApiRequest
} from '@src/classes';

export interface SocketMessageData {
    [key: string]: any
}

export interface SocketMessage {
    topic: string
    data: SocketMessageData
}

export class WSClient extends Emitter {

    private socket: WebSocket | null;
    private opening: boolean;
    private closed: boolean = false;

    constructor() {

        super();
    }

    /*
    **
    **
    */
    public open() {

        if (this.closed || this.opening || (this.socket && this.socket.readyState !== 3))
            return;

        this.opening = true;

        this.socket = new WebSocket(`wss://${location.hostname}/api/ws`, this.getReqStatId());
        
        this.socket.addEventListener('open', this.onOpen.bind(this));
        this.socket.addEventListener('close', this.onClose.bind(this));
        this.socket.addEventListener('message', this.onRawMessage.bind(this));
        this.socket.addEventListener('error', this.onError.bind(this));
    }

    /*
    **
    **
    */
    public close() : void {

        this.closed = true;

        this.socket?.close();
        this.socket = null;
    }

    /*
    **
    **
    */
    public send(topic: string, data: any = {}) {

        this.socket?.send(JSON.stringify({
            t: topic,
            data: data
        }));
    }
    
    /*
    **
    **
    */
    public canSend() : boolean {

        return this.socket && this.socket.readyState === 1 ? true : false;
    }

    /*
    **
    **
    */
    public getReqStatId() : string {

        let statId = localStorage.getItem(ApiRequest.LOCAL_STORAGE_TOKEN_NAME);

        if (statId === null)
            statId = Date.now().toString() + Math.random().toString();

        return Tools.sha256(`${statId}ws`);
    }

    /*
    **
    **
    */
    private async onOpen(event: Event) : Promise<void> {
        
        this.opening = false;

        this.emit('open');
    }

    /*
    **
    **
    */
    private async onClose(event: CloseEvent) : Promise<void> {

        this.socket = null;
        this.opening = false;

        this.emit('close');

        await Tools.sleep(2000);
        this.open();
    }

    /*
    **
    **
    */
    private onError(error: Error) : void {

        this.socket = null;
        this.opening = false;

        this.emit('error', error);
    }

    /*
    **
    **
    */
    private onRawMessage(event: MessageEvent) : void {

        let message: SocketMessage;

        try {
            message = JSON.parse(event.data)
        }
        catch(error) {
            console.error(error);
            return;
        }

        switch (message.topic) {

            /*case 'ping':
                this.send('pong');
                break;*/
            
            default: 
                this.emit(message.topic, message.data);
        }
    }
}