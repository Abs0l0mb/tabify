'use strict';

import { ServerResponse } from 'http';

export class Response extends ServerResponse {

    constructor(...args) {

        super(...args);
    }

    /*
    **
    **
    */
    isFinished() {

        return this.writableFinished || this.writableEnded;
    }
}