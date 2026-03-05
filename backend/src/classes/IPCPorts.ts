'use strict';

export class IPCPorts {

    /*
    **
    **
    */
    static mainProcess() : number {
        
        return process.env.MODE === 'development' ? 10_3_01 : 11_3_01;
    }

    /*
    **
    **
    */
    static subProcess() : number {
        
        return process.env.MODE === 'development' ? 10_3_02 : 11_3_02;
    }
}