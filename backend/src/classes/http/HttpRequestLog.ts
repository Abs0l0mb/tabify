'use strict';

import { 
    HttpRequest,
    HttpResponse,
    Log
} from '@src/classes';

export class HttpRequestLog {

    public constructor(private request: HttpRequest, private response: HttpResponse) {

        this.response.on('finish', () => {
            this.log();
        });
    }

    /*
    **
    **
    */
    private log() {

        Log.log(`${this.getHttpRequestLog()} ${this.getHttpResponseLog()}`);
    }

    /*
    **
    **
    */
    private getHttpRequestLog() : string {

        let remoteFamily = this.request.getRemoteFamily();
        let remoteAddress = this.request.getRemoteIp();
        let remotePort = this.request.getRemotePort();
        let remoteSocket = '';
        let method = this.request.method;
        let builtUrl = this.request.getURL();
        let url = '';
        let account = '';
        let parameters = this.getLightParameters(this.request.getParameters());

        if (remoteFamily === 'IPv6') {
            if (remoteAddress && remoteAddress.indexOf('::ffff:') === 0)
                remoteSocket = `${Log.Dim}[::ffff:${Log.Reset}${Log.Bright}${remoteAddress.slice(7)}${Log.Reset}${Log.Dim}]:${remotePort}${Log.Reset}`;
            else
                remoteSocket = `${Log.Dim}[${Log.Reset}${Log.Bright}${remoteAddress}${Log.Reset}${Log.Dim}]:${remotePort}${Log.Reset}`;
        }
        else
            remoteSocket = `${Log.FgGreen}${Log.Bright}${remoteAddress}${Log.Reset}${Log.FgGreen}${Log.Dim}:${remotePort}${Log.Reset}`;

        let software: string | null = null;
        let pua = this.request.getParsedUserAgent();
        let os = pua.os ? pua.os.name : '';
        let browser = pua.browser ? `${pua.browser.name} ${pua.browser.major}` : '';

        if (os && browser)
            software = `${Log.Dim}${os} ${browser}${Log.Reset}`;
            
        account = this.request.account ? `[${this.request.account.id}]` : '';
        method = `${Log.FgYellow}${Log.Bright}${method}${Log.Reset}`;
        url = `${Log.FgYellow}${Log.Bright}${builtUrl.pathname}${Log.Reset}${Log.FgYellow}${Log.Dim}${builtUrl.search}${Log.Reset}`;
        parameters = `${Log.FgYellow}${Log.Dim}${JSON.stringify(parameters)}${Log.Reset}`;
        
        return `${remoteSocket} ${software} ${account} ${method} ${url} ${parameters}`.trim();
    }

    /*
    **
    **
    */
    private getHttpResponseLog() : string {

        let statusCode = this.response.statusCode;
        let contentType = this.response.contentType;
        let data = this.response.data;
        let color = ![200, 204, 301, 302].includes(statusCode) || (data && data.error) ? Log.FgRed : Log.FgGreen;

        if (contentType)
            contentType = `${color}${contentType}${Log.Reset}`;

        let jsonSummary: string | null = null;
        if (statusCode === 200 && this.response.contentType === 'application/json') {
            if (data && data.error) {
                if (data.content)
                    jsonSummary = `${color}${Log.Bright}${data.content}${Log.Reset}`;
            }
        }

        let duration = `${Log.Dim}[${Date.now() - this.request.getTime()} ms]${Log.Reset}`;

        let output = `${Log.Dim}>${Log.Reset} ${color}${Log.Bright}${statusCode}${Log.Reset}`;

        if (contentType)
            output += ` ${contentType}`;
        if (jsonSummary)
            output += ` ${jsonSummary}`;

        output += ` ${duration}`;
        output += Log.Reset;

        return output;
    }

    /*
    **
    **
    */
    private getLightParameters(parameters: any) : any {

        let output: any = {};

        for (let key in parameters) {
            if (key.toLowerCase() === 'password')
                output[key] = '***';
            else if (typeof parameters[key] === 'object' && parameters[key] !== null) 
                output[key] = this.getLightParameters(parameters[key]);
            else if (parameters[key] === null)
                output[key] = null;
            else {

                let isNumber = typeof parameters[key] === 'number';

                let test = parameters[key].toString();

                if (test.length > 256)
                    test = test.slice(0, 256) + '...';
                
                if (isNumber)
                    test = new Number(test).valueOf();

                output[key] = test;
            }
        }

        return output;
    }
}