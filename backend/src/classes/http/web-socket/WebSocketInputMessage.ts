'use strict';

import { WebSocketClient } from '@src/classes';

export class WebSocketInputMessage {

    constructor(
        private client: WebSocketClient,
        private topic: string,
        private data: any
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
}