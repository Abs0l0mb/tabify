'use strict';

import { ApiRequest } from '@src/classes';

export interface IPendingRequests {
    [key: string]: ApiRequest
}

export class ApiRequestsManager {

    static pendingRequests: IPendingRequests = {};

    static locked: boolean = false;

    static async send(request: ApiRequest) {

        ApiRequestsManager.pendingRequests[request.id] = request;
        
        request.send();
    }

    /*
    **
    **
    */
    static clearPendingRequests() : void {

        for (const id in ApiRequestsManager.pendingRequests)
            ApiRequestsManager.clearPendingRequest(id);
    }

    /*
    **
    **
    */
    static clearPendingRequest(id: string) : void {

        if (this.locked)
            return;
        
        ApiRequestsManager.pendingRequests[id].abort(); 
        
        delete ApiRequestsManager.pendingRequests[id];
    }

    /*
    **
    **
    */
    static hasPendingRequests() : boolean {

        return Object.keys(ApiRequestsManager.pendingRequests).length > 0;
    }

    /*
    **
    **
    */
    static lock() : void {

        this.locked = true;
    }

    /*
    **
    **
    */
    static unlock() : void {

        this.locked = false;
    }
};