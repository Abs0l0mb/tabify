'use strict';

import { IPCResponseCallback } from '@src/classes';

export class IPCRequest {

    constructor(
        private topic: string, 
        private data: any, 
        private responseCallback?: IPCResponseCallback
    ) {}

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
    public respond(data: any = {}) : void {

        if (this.responseCallback)
            this.responseCallback(data);
        else
            throw new Error('IPC: No response callback registered');
    }
}