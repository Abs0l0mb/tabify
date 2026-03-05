'use strict';

import { AbstractClient } from '@src/classes';

export class ClientLocation {

    private static client: AbstractClient | any;

    /*
    **
    **
    */
    static set(client: AbstractClient | any) : void {
        
        ClientLocation.client = client;
    }

    /*
    **
    **
    */
    static get() : AbstractClient | any {

        return ClientLocation.client;
    }
};