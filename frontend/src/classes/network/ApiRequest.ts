'use strict';

import {
    Emitter,
    Api,
    Tools
} from '@src/classes';

export type TApiRequestMethod = 'GET' | 'POST';

export interface IApiRequestParameters {
    [key: string]: any
}

export interface IApiRequestData {
    method: TApiRequestMethod,
    endpoint: string,
    parameters: IApiRequestParameters,
    binary: boolean
}

export class ApiRequest extends Emitter {
    
    static readonly RESPONSE_TOKEN_NAME: string = 'x-stat';
    static readonly REQUEST_TOKEN_NAME: string = 'x-stat';
    static readonly LOCAL_STORAGE_TOKEN_NAME: string = 'build';
        
    public id: string = Tools.uid();
    private wrappedRequest: Request;
    private abortController: AbortController = new AbortController();

    constructor(private data: IApiRequestData) {

        super();

        this.initWrappedRequest();
    }

    /*
    **
    **
    */
    private initWrappedRequest() : void {

        const targetUrl = new URL(Api.getBaseURL());

        let endpoint = this.data.endpoint;

        if (targetUrl.pathname.slice(-1) === '/' && this.data.endpoint.slice(0, 1) === '/')
            endpoint = this.data.endpoint.slice(1);

        targetUrl.pathname += endpoint;

        const payload = ApiRequest.getURIEncodedPayload(this.data.parameters);

        if (this.data.method === 'GET')
            targetUrl.search = `?${payload}`;

        //this.abortController.signal.addEventListener("abort", () => {});

        this.wrappedRequest = new Request(targetUrl, {
            method: this.data.method,
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: this.data.method === 'POST' ? payload : undefined,
            signal: this.abortController.signal,
            credentials: 'include'
        });
    }

    /*
    **
    **
    */
    public async send() : Promise<void> {

        let response: Response;

        try {

            this.setRequestTokenHeader();

            response = await fetch(this.wrappedRequest);
            
            this.extractResponseTokenHeader(response);
        }
        catch(error) {
            return;
        }

        try {

            if (this.data.binary)
                return this.emit('success', await response.arrayBuffer());

            const data = await response.json();
    
            if (data.error === true)
                return this.emit('error', data.content);
            else
                return this.emit('success', data.content);
        }
        catch(error) {

            return this.emit('error', error);
        }
    }

    /*
    **
    **
    */
    public abort() : void {
        
        this.abortController.abort();
    }

    /*
    **
    **
    */
    static getURIEncodedPayload(parameters: IApiRequestParameters) : string {

        let payload: string[] = [];

        for (let property in parameters) {

            let value = parameters[property];

            if (value === undefined)
                continue;
            
            if (Array.isArray(value) || (typeof value === "object" && value !== null))
                value = JSON.stringify(value);

            const encodedPair = `${encodeURIComponent(property)}=${encodeURIComponent(value)}`;

            payload.push(encodedPair);
        }

        return payload.join('&');
    }

    /*
    **
    **
    */
    private setRequestTokenHeader() : void {

        let token = localStorage.getItem(ApiRequest.LOCAL_STORAGE_TOKEN_NAME);

        if (token === null)
            token = Tools.uid();
        
        this.wrappedRequest.headers.set(ApiRequest.REQUEST_TOKEN_NAME, this.hashToken(token));
    }

    /*
    **
    **
    */
    private extractResponseTokenHeader(response: Response) : void {

        const token = response.headers.get(ApiRequest.RESPONSE_TOKEN_NAME);

        if (token)
            localStorage.setItem(ApiRequest.LOCAL_STORAGE_TOKEN_NAME, this.hashToken(token));
    }

    /*
    **
    **
    */
    private hashToken(token: string) : string {
        
        return Tools.sha256(`${token}ws`);
    }
}