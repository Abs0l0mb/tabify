'use strict';

import * as http from 'http';

export class HttpRequest extends http.IncomingMessage {

    constructor(...args) {

        super(...args);

        this.time = Date.now();
    }

    /*
    **
    **
    */
    getRemoteIp() {

        const realIp = this.headers['x-real-ip'];

        if (realIp)
            return realIp;

        return this.socket.remoteAddress ? this.socket.remoteAddress : null;
    }

    /*
    **
    **
    */
    getRemoteFamily() {

        if (/^((25[0-5]|(2[0-4]|1[0-9]|[1-9]|)[0-9])(\.(?!$)|$)){4}$/.test(this.getRemoteIp()))
            return 'IPv4';
        else
            return 'IPv6';
    }

    /*
    **
    **
    */
    decomposeUrl() {
        
        try {

            this.builtUrl = new URL(this.url, `http://${this.headers['host']}`);
        }
        catch(error) {

            this.builtUrl = new URL(this.url, 'undefined://-');
        }

        if (this.builtUrl.pathname.slice(-1) === '/')
            this.builtUrl.pathname = this.builtUrl.pathname.slice(0, -1);

        if (this.builtUrl.search.length > 200)
            this.builtUrl.search = `${this.builtUrl.search.slice(0, 200)}...`;
    }
}