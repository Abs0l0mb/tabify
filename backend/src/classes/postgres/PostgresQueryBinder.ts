'use strict';

export class PostgresQueryBinder {

    private i: number = 0;
    private params: any[] = [];

    /*
    **
    **
    */
    public addParam(value: any) : string {

        this.i++;

        const key = this.i;

        if (value === undefined)
            value = '';
            
        this.params.push(value);

        return `$${key}`;
    }

    /*
    **
    **
    */
    public getParams() : any[] {

        return this.params;
    }
}