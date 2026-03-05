'use strict';

import { WebSocketClient } from '@src/classes';

import { WebSocket } from 'ws';

export class WebSocketInputRequest {

    private start: number = Date.now();
    private end: number = Date.now();

    private finished: boolean = false;
    private error: string | null = null;

    constructor(
        private client: WebSocketClient,
        private topic: string, 
        private data: any,
        private id: string
    ) {}

    /*
    **
    **
    */
    public getClient() : WebSocketClient {

        return this.client;
    }

    /*
    **
    **
    */
    public getTopic() : string {

        return this.topic;
    }

    /*
    **
    **
    */
    public getData() : any {

        return this.data;
    }

    /*
    **
    **
    */
    private respond(data: any = {}) : Promise<void> {

        return new Promise<void>((resolve, reject) => {

            if (this.finished) {
                reject(new Error('Request finished'));
                return;
            }

            this.end = Date.now();
            this.finished = true;

            if (this.client.getSocket().readyState === WebSocket.OPEN) {

                this.client.getSocket().send(JSON.stringify({
                    id: this.id,
                    data: data
                }), {}, () => {
                    resolve();
                });
            }
            else
                reject(new Error(`Cannot respond due to socket not opened`));
        });
    }

    /*
    **
    **
    */
    public respondSuccessContent(content: any = true) : Promise<void> {

        return this.respond({
            content: content
        });
    }

    /*
    **
    **
    */
    public respondErrorContent(content: any = true) : Promise<void> {

        if (content instanceof Error)
            content = content.message;
        else if (typeof content !== 'string')
            content = "request-error";
        
        this.error = content;

        return this.respond({
            error: true,
            content: content
        });
    }
    
    /*
    **
    **
    */
    public isFinished() : boolean {

        return this.finished;
    }

    /*
    **
    **
    */
    public getDuration() : number {
        
        return this.end - this.start;
    }

    /*
    **
    **
    */
    public getError() : string | null {
        
        return this.error;
    }
}