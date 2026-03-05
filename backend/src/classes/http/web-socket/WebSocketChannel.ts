'use strict';

import {
    WebSocketClient,
    WebSocketInputRequest,
    WebSocketInputMessage,
    PublicError,
    Log
}from '@src/classes';

import { WebSocket } from 'ws';

export interface WebSocketMessage {
    readonly topic: string
    readonly data: any
}

export interface WebSocketRawInput {
    topic?: string,
    data?: any
    id?: string,
}

export type WebSocketInputRequestCallback = (request: WebSocketInputRequest) => Promise<void>;
export type WebSocketInputMessageCallback = (message: WebSocketInputMessage) => Promise<void>;

export class WebSocketChannel {

    private requestCallbacks: Map<string, WebSocketInputRequestCallback> = new Map();
    private messageCallbacks: Map<string, WebSocketInputMessageCallback> = new Map();

    constructor(private key: string, private builtInController: boolean = false) {}

    public readonly clients: Map<string, WebSocketClient> = new Map();
    
    /*
    **
    **
    */
    public getKey() : string {

        return this.key;
    }

    /*
    **
    **
    */
    public addClient(client: WebSocketClient) : WebSocketClient {

        this.onClientJoin(client);

        client.on('close', () => {
    
            this.removeClient(client);
        });

        client.on('input', (buffer: Buffer) => {
            this.onClientInput(client, buffer);
        });

        this.clients.set(client.getKey(), client);

        client.log(`JOINED ${this.key}`);

        return client;
    }
    
    /*
    **
    **
    */
    public removeClient(client: WebSocketClient) : void {

        const stored = this.clients.get(client.getKey());

        if (stored) {

            this.clients.delete(client.getKey())

            this.onClientLeave(client);
            
            client.log(`LEAVED ${this.key}`);
        }
    }

    /*
    **
    **
    */
    public broadcast(message: WebSocketMessage) : void {

        for (const client of this.clients.values()) {

            if (client.getSocket().readyState === WebSocket.OPEN)
                client.getSocket().send(JSON.stringify(message));
        }
    }

    /*
    **
    **
    */
    private onClientInput(client: WebSocketClient, buffer: Buffer) : void {

        if (!this.builtInController)
            return;
        
        const data: any = JSON.parse(buffer.toString('utf-8'));

        if (!data.hasOwnProperty('topic')
         || !(typeof data.topic === 'string')
         || !data.topic.startsWith('/')
         || !data.hasOwnProperty('data'))
            return;

        if (data.hasOwnProperty('id') && typeof data.id === 'string')
            this.onRequest(new WebSocketInputRequest(client, data.topic, data.data, data.id));
        else
            this.onMessage(new WebSocketInputMessage(client, data.topic, data.data));
    }

    /*
    **
    **
    */
    protected async onRequest(request: WebSocketInputRequest) : Promise<void> {
        
        try {
            
            if (!(await this.lookForRequestCallback(request)))
                throw new PublicError('request-callback-not-found');

            if (!request.isFinished())
                throw new Error('Request not finished');

            request.getClient().logRequest(request);
        }
        catch(error) {

            if (typeof error === 'object'
                && error !== null
                && error.public
                && error.message) {
                
                request.respondErrorContent(error.message);
            }
            else {

                request.respondErrorContent('internal-error');

                Log.red(`Error when handling request ${request.getTopic()}`);
                Log.printError(error);
            }

            request.getClient().logRequest(request);
        }
    }

    /*
    **
    **
    */
    protected async onMessage(message: WebSocketInputMessage) : Promise<void> {
        
        try {
            
            if (!this.lookForMessageCallback(message))
                throw new Error('message-callback-not-found');
        }
        catch(error) {

            Log.red(`Error when handling message ${message.getTopic()}`);
            Log.printError(error);
        }
    }

    /*
    **
    **
    */
    protected async lookForRequestCallback(request: WebSocketInputRequest) : Promise<boolean> {

        for (const [topic, requestCallback] of this.requestCallbacks) {

            if (topic === request.getTopic()) {

                await requestCallback(request);

                return true;
            }
        }

        return false;
    }

    /*
    **
    **
    */
    protected lookForMessageCallback(message: WebSocketInputMessage) : boolean {
        
        for (const [topic, messageCallback] of this.messageCallbacks) {

            if (topic === message.getTopic()) {

                message.getClient().logMessage(message);

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
    public request(topic: string, callback: WebSocketInputRequestCallback) {

        this.requestCallbacks.set(topic, callback);
    }

    /*
    **
    **
    */
    public message(topic: string, callback: WebSocketInputMessageCallback) {

        this.messageCallbacks.set(topic, callback);
    }

    /*
    **
    **
    */
    protected onClientJoin(client: WebSocketClient) : void {};

    /*
    **
    **
    */
    protected onClientLeave(client: WebSocketClient) : void {};
}