'use strict';

import {
    WebSocketClient,
    Emitter,
    Listener,
    Tools,
    WebSocketRawInput
} from "@src/classes";

export interface WebSocketRawOutputRequest {
    readonly topic: string,
    readonly data: any,
    readonly id: string
}

export class WebSocketOutputRequest extends Emitter {

    private id: string = Tools.uid(32);
    private listener: Listener;

    constructor(private topic: string, private data: any, private client: WebSocketClient) {

        super();

        this.listener = client.on('input', this.onInput.bind(this));
    }

    /*
    **
    **
    */
    public send() : void {

        const message: WebSocketRawOutputRequest = {
            topic: this.topic,
            data: this.data,
            id: this.id
        };

        try {
            if (this.client.isOpened())
                this.client.getSocket()?.send(JSON.stringify(message));
        } catch(error) {
            console.error(error);
        }
    }

    /*
    **
    **
    */
    private onInput(input: WebSocketRawInput) : void {

        if (!input.hasOwnProperty('data')
         || !input.hasOwnProperty('id')
         || input.id !== this.id)
            return;

        this.listener.off();

        this.emit('response', input.data);
    }
}