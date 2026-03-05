'use strict';

import {
    Block,
    Div,
    DataTable,
    Tools
} from '@src/classes';

export interface DataTableHeadCellSearchData {
    text: string;
}

export interface DataTableHeadCellSortData {
    method: 'SORT_DISABLED' | 'ASC' | 'DESC';
}

export class DataTableHeadCell extends Block {

    private labelContainer: Div;
    private searchButton: Div;
    private searchContainer: Div;
    private searchInput: Block;
    private searchTimeout: any;

    constructor(private column: string, parent: Block) {
    
        super('td', {}, parent);

        this.labelContainer = new Div('label-container', this);
        this.searchButton = new Div('search-button', this.labelContainer);

        new Div('label', this.labelContainer).write(this.beautifyCamelCase(column));
        new Div('sort', this.labelContainer);
        
        this.searchContainer = new Div('search-container', this);

        this.searchInput = new Block('input', {
            type: 'text',
            placeholder: 'Search'
        }, this.searchContainer);
        
        this.setData('method', DataTable.SORT_DISABLED);

        this.onNative('click', this.onSort.bind(this));

        this.searchButton.onNative('click', this.onSearchButton.bind(this));

        this.searchInput.onNative('input', this.onSearch.bind(this));
        this.searchInput.onNative('click', (event: Event) => { event.stopPropagation(); });
    }

    /*
    **
    **
    */
    private beautifyCamelCase(camelCaseString: string): string {

        const spacedString = camelCaseString.replace(/([a-z])([A-Z])/g, '$1 $2');
        
        return spacedString.charAt(0).toUpperCase() + spacedString.slice(1);
    }

    /*
    **
    **
    */
    private async onSearchButton(event: Event) : Promise<void> {

        event.stopPropagation();

        if (parseInt(this.getData('search-displayed')) === 1)
            this.setData('search-displayed', 0);
        else
            this.setData('search-displayed', 1);

        if (this.searchInput.element.value !== '') {
            this.searchInput.element.value = '';
            this.emitSearchData();
        }

        await Tools.sleep(5);
        this.searchInput.element.focus();
    }

    /*
    **
    **
    */
    private onSearch(event: Event) : void {

        clearTimeout(this.searchTimeout);

        this.searchTimeout = setTimeout(this.emitSearchData.bind(this), 60);
    }

    /*
    **
    **
    */
    private onSort() : void {

        let oldMethod = this.getData('method');
        let newMethod = DataTable.SORT_DISABLED;

        if (oldMethod === DataTable.SORT_DISABLED)
            newMethod = DataTable.SORT_ASCENDING;
        else if (oldMethod === DataTable.SORT_ASCENDING)
            newMethod = DataTable.SORT_DESCENDING;
        
        this.setData('method', newMethod);
        
        this.emitSortData();
    }

    /*
    **
    **
    */
    private emitSearchData() : void {

        this.emit('search-data', {
            text: this.searchInput.element.value
        });

        this.emit('resize');
    }

    /*
    **
    **
    */
    private emitSortData() : void {

        this.emit('sort-data', {
            method: this.getData('method')
        });

        this.emit('resize');
    }

    /*
    **
    **
    */
    public resetSort() : void {

        this.setData('method', DataTable.SORT_DISABLED);

        this.emit('resize');
    }

    /*
    **
    **
    */
    public getColumn() : string {

        return this.column;
    }
}