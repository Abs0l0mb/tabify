'use strict';

import { UAParser } from 'ua-parser-js';
import { Log } from '../Log.js';

export class RequestLogger {

    constructor(request, response) {

        this.request = request;
        this.response = response;

        this.response.on('finish', () => {

            this.log();
        });
    }

    /*
    **
    **
    */
    log() {

        Log.log(`${this.getRequestLog()} ${this.getResponseLog()}`);
    }

    /*
    **
    **
    */
    getRequestLog() {

        let remoteFamily = this.request.getRemoteFamily();
        let remoteAddress = this.request.getRemoteIp();
        let remotePort = this.request.socket.remotePort;
        let remoteSocket = '';
        let method = this.request.method;
        let url = this.request.builtUrl;
        let parameters = this.request.parameters ? this.request.parameters : {};

        if (remoteFamily === 'IPv6') {
            if (remoteAddress.indexOf('::ffff:') === 0)
                remoteSocket = `${Log.Dim}[::ffff:${Log.Reset}${Log.Log.Bright}${remoteAddress.slice(7)}${Log.Reset}${Log.Dim}]:${remotePort}${Log.Reset}`;
            else
                remoteSocket = `${Log.Dim}[${Log.Reset}${Log.Bright}${remoteAddress}${Log.Reset}${Log.Dim}]:${remotePort}${Log.Reset}`;
        }
        else
            remoteSocket = `${Log.FgGreen}${Log.Bright}${remoteAddress}${Log.Reset}${Log.FgGreen}${Log.Dim}:${remotePort}${Log.Reset}`;

        let software = null;
        let ua = UAParser(this.request.headers['user-agent']);
        let os = ua.os ? ua.os.name : '';
        let browser = ua.browser ? `${ua.browser.name} ${ua.browser.major}` : '';
        if (os && browser)
            software = `${Log.Dim}${os} ${browser}${Log.Reset}`;
            
        method = `${Log.FgYellow}${Log.Bright}${method}${Log.Reset}`;
        url = `${Log.FgYellow}${Log.Bright}${url.pathname}${Log.Reset}${Log.FgYellow}${Log.Dim}${url.search}${Log.Reset}`;

        return `${remoteSocket} ${software} ${method} ${url}`.trim();
    }

    /*
    **
    **
    */
    getResponseLog() {

        let statusCode = this.response.statusCode;
        let contentType = this.response.contentType;
        let data = this.response.data;
        let color = ![200, 301, 302].includes(statusCode) || (data && data.error) ? Log.FgRed : Log.FgGreen;

        if (contentType)
            contentType = `${color}${contentType}${Log.Reset}`;

        let jsonSummary = null;
        if (statusCode === 200 && this.response.contentType === 'application/json') {
            if (data && data.error) {
                if (data.content)
                    jsonSummary = `${color}${Log.Bright}${data.content}${Log.Reset}`;
            }
        }

        let duration = `${Log.Dim}[${Date.now() - this.request.time} ms]${Log.Reset}`;

        let output = `${Log.Dim}>${Log.Reset} ${color}${Log.Bright}${statusCode}${Log.Reset}`;
        if (contentType)
            output += ` ${contentType}`;
        if (jsonSummary)
            output += ` ${jsonSummary}`;
        output += ` ${duration}`;
        output += Log.Reset;

        return output;
    }
}