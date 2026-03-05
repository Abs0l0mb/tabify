'use strict';

import * as http from 'http';

export class HttpResponse extends http.ServerResponse {

    constructor(...args) {

        super(...args);
    }
}