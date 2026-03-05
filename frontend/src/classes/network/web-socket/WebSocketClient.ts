'use strict';

import {
    Tools,
    Emitter,
    ApiRequest,
    WebSocketOutputRequest
} from '@src/classes';

export interface WebSocketRawInput {
    topic?: string,
    data?: any
    id?: string,
}

export interface WebSocketMessage {
    readonly topic: string
    readonly data: any
}

export type WebSocketInputMessageCallback = (message: WebSocketMessage) => Promise<void>;

export class WebSocketClient extends Emitter {

    private socket: WebSocket | null;
    private opening: boolean;
    private closed: boolean = false;

    private messageCallbacks: Map<string, WebSocketInputMessageCallback> = new Map();

    constructor(private path: string) {

        super();
    }

    /*
    **
    **
    */
    public async open() : Promise<void> {

        return new Promise(resolve => {

            if (this.closed || this.opening || (this.socket && this.socket.readyState !== 3))
                return;

            this.messageCallbacks = new Map();

            this.opening = true;

            this.socket = new WebSocket(`wss://${location.hostname}/api${this.path}`, this.getReqStatId());
            
            this.socket.addEventListener('open', (event: Event) => {
                resolve();
                this.onOpen(event);
            });

            this.socket.addEventListener('close', this.onClose.bind(this));
            this.socket.addEventListener('message', this.onInput.bind(this));
            this.socket.addEventListener('error', this.onError.bind(this));
        });
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
    public sendMessage(topic: string, data: any = {}) : void {

        const message: WebSocketMessage = {
            topic: topic,
            data: data
        };

        try {
            if (this.isOpened())
                this.socket?.send(JSON.stringify(message));
        } catch(error) {
            console.error(error);
        }
    }

    /*
    **
    **
    */
    public sendRequest(topic: string, data: any = {}) : Promise<any> {

        return new Promise((resolve, reject) => { 

            const request =  new WebSocketOutputRequest(topic, data, this);

            request.on('response', (data: any) => {
                
                if (data.hasOwnProperty('error') && data.error === true)
                    reject(data.content);
                else
                    resolve(data.content);
            });

            request.send();
        });
    }
    
    /*
    **
    **
    */
    public isOpened() : boolean {

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
    private onInput(event: MessageEvent) : void {

        let input: any;

        try {
            input = JSON.parse(event.data)
        }
        catch(error) {
            console.error(error);
            return;
        }

        if (input.hasOwnProperty('data')
         && input.hasOwnProperty('id')) {
            
            this.emit('input', input)
        }
        
        else if (input.hasOwnProperty('data')) {

            const message: WebSocketMessage = {
                topic: input.topic,
                data: input.data
            };

            this.onMessage(message);
        }
    }

    /*
    **
    **
    */
    protected async onMessage(message: WebSocketMessage) : Promise<void> {
        
        try {
            
            if (!this.lookForMessageCallback(message))
                throw new Error(`Message callback not found for topic ${message.topic}`);
        }
        catch(error) {

            console.log(`Error when handling message topic ${message.topic}`, error);
        }
    }

    /*
    **
    **
    */
    protected lookForMessageCallback(message: WebSocketMessage) : boolean {
        
        for (const [topic, messageCallback] of this.messageCallbacks) {

            if (topic === message.topic) {

                messageCallback(message);
                
                return true;
            }
        }

        return false;
    }

    /*
    **
    **
    */
    public listen(topic: string, callback: WebSocketInputMessageCallback) : void {

        this.messageCallbacks.set(topic, callback);
    }

    /*
    **
    **
    */
    public getSocket() : WebSocket | null {

        return this.socket;
    }
}